from __future__ import annotations

import pytest
import torch

from model.architecture import HybridTTTLoopedCoder
from model.config import HybridTTTConfig
from model.moe import FineGrainedMoE
from model.sparse_attention import NativeSparseAttention
from model.ttt_layer import TTTLinearLayer


def test_full_16b_config_parameter_budget_and_validation() -> None:
    full_cfg = HybridTTTConfig.full_16b()
    counts = full_cfg.estimate_parameter_counts()
    total_b = counts["total_params"] / 1e9
    active_b = counts["active_params_per_token"] / 1e9

    assert 15.0 <= total_b <= 17.0
    assert 2.4 <= active_b <= 3.1
    assert full_cfg.max_position_embeddings == 1_048_576
    assert 0.80 <= full_cfg.ttt_ratio <= 0.90

    with pytest.raises(ValueError):
        HybridTTTConfig(hidden_size=65, num_heads=4)
    with pytest.raises(ValueError):
        HybridTTTConfig(moe_num_experts=8, moe_top_k=16)


def test_ttt_linear_strict_causality_and_state_continuity() -> None:
    torch.manual_seed(7)
    cfg = HybridTTTConfig.tiny_test()
    layer = TTTLinearLayer(cfg).eval()

    x = torch.randn(2, 37, cfg.hidden_size)
    x_perturbed = x.clone()
    # Mutate strictly future positions t >= 20
    x_perturbed[:, 20:, :] += 5.0

    with torch.no_grad():
        y1, state1 = layer(x)
        y2, _ = layer(x_perturbed)

    assert y1.shape == x.shape
    assert state1.weight.shape == (2, cfg.num_heads, cfg.head_dim, cfg.head_dim)
    assert state1.bias.shape == (2, cfg.num_heads, 1, cfg.head_dim)
    # Prefix t < 20 must remain bit-for-bit identical (zero future leakage)
    assert torch.allclose(y1[:, :20, :], y2[:, :20, :], atol=1e-6, rtol=1e-5)
    # Suffix t >= 20 must diverge
    assert not torch.allclose(y1[:, 20:, :], y2[:, 20:, :], atol=1e-3)

    # Chunk-aligned streaming continuity test (split exactly at chunk boundary = 16)
    with torch.no_grad():
        y_part1, st_mid = layer(x[:, :16, :])
        y_part2, st_end = layer(x[:, 16:, :], state=st_mid)
        y_stream = torch.cat((y_part1, y_part2), dim=1)
    assert torch.allclose(y1, y_stream, atol=1e-5, rtol=1e-5)
    assert torch.allclose(state1.weight, st_end.weight, atol=1e-5, rtol=1e-5)


def test_native_sparse_attention_strict_causality_and_short_sequence() -> None:
    torch.manual_seed(11)
    cfg = HybridTTTConfig.tiny_test()
    nsa = NativeSparseAttention(cfg).eval()

    # Multi-block sequence (seq_len=40 > block_size=8)
    x = torch.randn(2, 40, cfg.hidden_size)
    x_mod = x.clone()
    x_mod[:, 25:, :] += 7.5

    with torch.no_grad():
        y1 = nsa(x)
        y2 = nsa(x_mod)

    assert y1.shape == x.shape
    assert torch.isfinite(y1).all()
    assert torch.allclose(y1[:, :25, :], y2[:, :25, :], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(y1[:, 25:, :], y2[:, 25:, :], atol=1e-3)

    # Short sequence within single block (seq_len < block_size): must not NaN in forward/backward
    nsa.train()
    x_short = torch.randn(2, 5, cfg.hidden_size, requires_grad=True)
    y_short = nsa(x_short)
    loss = y_short.pow(2).mean()
    loss.backward()
    assert torch.isfinite(y_short).all()
    assert x_short.grad is not None and torch.isfinite(x_short.grad).all()


def test_moe_64_experts_top6_and_load_balancing_loss() -> None:
    torch.manual_seed(19)
    cfg = HybridTTTConfig(
        vocab_size=256,
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        num_prelude_layers=1,
        num_core_layers=8,
        num_coda_layers=1,
        core_layer_pattern=("ttt", "ttt", "ttt", "ttt", "ttt", "ttt", "ttt", "sparse_attn"),
        moe_num_experts=64,
        moe_top_k=6,
        moe_intermediate_size=16,
        shared_intermediate_size=32,
    )
    moe = FineGrainedMoE(cfg)
    x = torch.randn(2, 16, cfg.hidden_size, requires_grad=True)
    out = moe(x)

    assert out.hidden_states.shape == x.shape
    assert out.aux_loss.ndim == 0
    assert float(out.aux_loss.item()) > 0.0
    assert int(out.expert_counts.sum().item()) == 2 * 16 * 6

    (out.hidden_states.pow(2).mean() + out.aux_loss).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.router.weight.grad is not None and torch.isfinite(moe.router.weight.grad).all()


def test_truncated_bptt_gradient_flow_and_early_halting() -> None:
    torch.manual_seed(23)
    cfg = HybridTTTConfig.tiny_test()
    model = HybridTTTLoopedCoder(cfg)

    # 1. Verify Truncated BPTT (6 steps, truncated window = 2):
    #    Prelude, Core, Router, and Coda must all receive finite gradients.
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    out = model(ids, labels=ids, num_steps=6, adaptive_halt=False)
    assert out.loss is not None
    assert out.steps_used == 6
    assert out.logits.shape == (2, 24, cfg.vocab_size)
    out.loss.backward()

    assert model.token_embeddings.weight.grad is not None
    assert torch.isfinite(model.token_embeddings.weight.grad).all()
    assert float(model.token_embeddings.weight.grad.abs().sum().item()) > 0.0

    # Early-step embeddings (step 0) live strictly outside the truncated BPTT window (steps 4, 5)
    step_emb_grad = model.recurrent_core.input_injection.step_embed.weight.grad
    assert step_emb_grad is not None
    assert float(step_emb_grad[0].abs().sum().item()) == 0.0
    assert float(step_emb_grad[5].abs().sum().item()) > 0.0

    # 2. Verify Adaptive Compute Router early halting during inference
    model.eval()
    # Bias router strongly positive to force immediate halt at min_recurrent_steps
    final_linear = model.recurrent_core.compute_router.proj[2]
    assert isinstance(final_linear, torch.nn.Linear) and final_linear.bias is not None
    with torch.no_grad():
        final_linear.bias.fill_(10.0)

    with torch.no_grad():
        halt_out = model(ids, adaptive_halt=True)
    assert halt_out.steps_used == cfg.min_recurrent_steps
    assert (halt_out.per_sample_steps == float(cfg.min_recurrent_steps)).all()

    # 3. Verify generation returns expected shapes
    gen_ids, trace = model.generate(ids[:1, :8], max_new_tokens=6, temperature=0.7)
    assert gen_ids.shape == (1, 14)
    assert len(trace) == 6
