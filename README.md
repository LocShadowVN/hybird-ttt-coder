# Hybrid-TTT-Coder

**Hybrid-TTT-Coder** is an open-source, code-specialized hybrid neural + systems architecture (~16B total / ~2.6B active parameters per token, 1M-token context window) combining:

- **PyTorch Neural Core (`model/`)**:
  - **TTT-Linear (`model/ttt_layer.py`)**: Chunked dual-form Test-Time Training with O(N) time complexity and $O(1)$ fast-weight state memory.
  - **Native Sparse Attention (`model/sparse_attention.py`)**: Strictly causal compressed block attention, Top-$K$ selected fine-grained block attention, and $O(T \cdot W)$ chunked sliding window attention with RoPE ($\theta = 10^6$).
  - **Fine-Grained MoE (`model/moe.py`)**: 64 routed SwiGLU experts (Top-6 active) + 1 shared expert with load-balancing and router z-loss.
  - **Looped Recurrent Core (`model/recurrent_core.py`)**: Weight-tied 12-layer core (~85% TTT / ~15% Sparse Attention) with Input Injection, Adaptive Compute Router ($K \in [1, 16]$), and Truncated BPTT.
- **Rust Native Systems Engine (`rust_core/` -> `hybrid_ttt_native`)**:
  - **Parallel Sandbox Verifier (`sandbox_verifier.rs`)**: Process-group isolated compilation, linting, and unit-test runner with `RLIMIT_AS`/`RLIMIT_CPU` guards, bounded non-blocking stdout/stderr capture, hard timeout enforcement, objective shaped reward scoring, and GIL-free Rayon batch parallelism.
  - **Parallel Git History Extractor (`git_history_extractor.rs`)**: Thread-local `git2::Repository` workers via `rayon::map_init`, filtering binaries, lockfiles, and minified assets to emit clean `(Commit / Context -> Code Diff -> Unit Test Diff)` JSONL datasets.
- **Self-Evolution & Training (`runtime/`, `train/`, `eval/`)**:
  - Graph Memory (`SkillLibrary`) + 4-stage `Diff -> Mutate -> Verify -> Select` engine (`EvolutionaryLoop`).
  - SFT distillation from Rust-extracted Git JSONL (`train/sft_distill.py`) and critic-free RLVR-GRPO (`train/rlvr_grpo.py`).

## Quickstart

```bash
# 1. Create virtual environment & install Python dependencies (CPU PyTorch example)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3.0"
pip install -e ".[dev]" --no-build-isolation

# 2. Build Rust native PyO3 extension (`hybrid_ttt_native`)
maturin develop --release

# 3. Run Rust tests + Python unit tests + Coverage + Quantitative Benchmark
make rust-test
make py-lint
make py-test
make eval
