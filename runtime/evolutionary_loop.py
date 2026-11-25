from __future__ import annotations

import ast
import difflib
import hashlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import hybrid_ttt_native as native
from runtime.skill_library import SkillLibrary, SkillNode


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    candidate_id: str
    module_filename: str
    source_code: str
    mutation_name: str
    parent_id: str | None = None
    consulted_skill_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BottleneckDiff:
    unified_diff: str
    max_loop_depth: int
    recursive_functions: tuple[str, ...]
    quadratic_membership_loops: int
    string_concat_in_loops: int
    bottlenecks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    generations: int = 2
    mutations_per_generation: int = 3
    timeout_ms: int = 3_000
    memory_limit_mb: int = 1_024
    num_threads: int | None = None


@dataclass(frozen=True, slots=True)
class EvolutionStepReport:
    generation: int
    diff_report: BottleneckDiff
    evaluated_count: int
    passed_count: int
    selected_candidate: ArchitectureCandidate
    selected_result: native.VerificationResult
    improved: bool
    learned_skill_id: str | None


class MutationGenerator(Protocol):
    def generate(
        self,
        parent: ArchitectureCandidate,
        diff_report: BottleneckDiff,
        retrieved_skills: list[SkillNode],
        max_mutations: int,
    ) -> list[ArchitectureCandidate]: ...


def analyze_source_bottlenecks(baseline_code: str, current_code: str) -> BottleneckDiff:
    """Step 1 (Diff): Compare old and new architectures via unified diff and static AST
    bottleneck analysis (recursion, nested loop depth, O(N^2) lookups, string concat)."""
    unified = "".join(
        difflib.unified_diff(
            baseline_code.splitlines(keepends=True),
            current_code.splitlines(keepends=True),
            fromfile="baseline.py",
            tofile="current.py",
        )
    )

    max_loop_depth = 0
    recursive_fns: list[str] = []
    quadratic_lookups = 0
    str_concat_loops = 0

    try:
        tree = ast.parse(current_code)
    except SyntaxError:
        return BottleneckDiff(
            unified_diff=unified,
            max_loop_depth=0,
            recursive_functions=(),
            quadratic_membership_loops=0,
            string_concat_in_loops=0,
            bottlenecks=("syntax_error",),
        )

    def walk_depth(node: ast.AST, loop_depth: int, fn_name: str | None) -> None:
        nonlocal max_loop_depth, quadratic_lookups, str_concat_loops
        next_depth = loop_depth
        next_fn = fn_name

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            next_fn = node.name
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            next_depth = loop_depth + 1
            max_loop_depth = max(max_loop_depth, next_depth)
        elif isinstance(node, ast.Call) and next_fn is not None:
            if isinstance(node.func, ast.Name) and node.func.id == next_fn:
                if next_fn not in recursive_fns:
                    recursive_fns.append(next_fn)
        elif isinstance(node, ast.Compare) and loop_depth >= 1:
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                quadratic_lookups += 1
        elif isinstance(node, ast.AugAssign) and loop_depth >= 1:
            if isinstance(node.op, ast.Add) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                str_concat_loops += 1

        for child in ast.iter_child_nodes(node):
            walk_depth(child, next_depth, next_fn)

    walk_depth(tree, 0, None)

    bottlenecks: list[str] = []
    if recursive_fns:
        bottlenecks.append("unmemoized_recursion")
    if max_loop_depth >= 2:
        bottlenecks.append("nested_loop_complexity")
    if quadratic_lookups > 0:
        bottlenecks.append("linear_membership_scan_in_loop")
    if str_concat_loops > 0:
        bottlenecks.append("string_concat_in_loop")
    if not bottlenecks:
        bottlenecks.append("general_micro_optimization")

    return BottleneckDiff(
        unified_diff=unified,
        max_loop_depth=max_loop_depth,
        recursive_functions=tuple(recursive_fns),
        quadratic_membership_loops=quadratic_lookups,
        string_concat_in_loops=str_concat_loops,
        bottlenecks=tuple(bottlenecks),
    )


class HeuristicCodeMutator:
    """Deterministic AST-safe code mutator guided by bottleneck analysis and Graph Memory."""

    def generate(
        self,
        parent: ArchitectureCandidate,
        diff_report: BottleneckDiff,
        retrieved_skills: list[SkillNode],
        max_mutations: int,
    ) -> list[ArchitectureCandidate]:
        skill_ids = tuple(s.skill_id for s in retrieved_skills)
        variants: list[tuple[str, str]] = []
        src = parent.source_code

        # Mutation 1: Inject functools.lru_cache onto unmemoized recursive functions
        if "unmemoized_recursion" in diff_report.bottlenecks and "lru_cache" not in src:
            memoized = self._inject_lru_cache(src, diff_report.recursive_functions)
            if memoized != src:
                variants.append(("memoize_recursion_lru_cache", memoized))

        # Mutation 2: Convert list-backed membership scan to O(1) set lookup
        if "linear_membership_scan_in_loop" in diff_report.bottlenecks and "seen = []" in src:
            set_refactored = src.replace("seen = []", "seen = set()").replace(
                "seen.append(", "seen.add("
            )
            if set_refactored != src:
                variants.append(("hash_set_membership_lookup", set_refactored))

        # Mutation 3: Replace O(N^2) string concatenation in loops with list + join
        if "string_concat_in_loop" in diff_report.bottlenecks and 'result = ""' in src:
            joined = (
                src.replace('result = ""', "parts: list[str] = []")
                .replace("result += ", "parts.append(")
                .replace("return result", 'return "".join(parts)')
            )
            # Close appended parentheses cleanly line-by-line
            fixed_lines: list[str] = []
            for line in joined.splitlines():
                if "parts.append(" in line and not line.rstrip().endswith(")"):
                    fixed_lines.append(line + ")")
                else:
                    fixed_lines.append(line)
            joined_src = "\n".join(fixed_lines) + "\n"
            if self._is_valid_python(joined_src) and joined_src != src:
                variants.append(("linear_string_join_buffer", joined_src))

        # Mutation 4: Apply verified skill code pattern hint as clean wrapper or fallback
        for skill in retrieved_skills:
            if skill.code_pattern and skill.code_pattern not in src:
                header_annotated = f"# Applied skill: {skill.skill_id}\n{src}"
                variants.append((f"skill_guided_{skill.skill_id}", header_annotated))
                break

        # Ensure at least one identity challenger is evaluated to measure baseline variance
        if not variants:
            variants.append(("baseline_preserve", src))

        candidates: list[ArchitectureCandidate] = []
        for idx, (mut_name, code) in enumerate(variants[:max_mutations]):
            digest = hashlib.sha256(f"{parent.candidate_id}:{mut_name}:{code}".encode()).hexdigest()[
                :10
            ]
            candidates.append(
                ArchitectureCandidate(
                    candidate_id=f"cand_{idx}_{digest}",
                    module_filename=parent.module_filename,
                    source_code=code,
                    mutation_name=mut_name,
                    parent_id=parent.candidate_id,
                    consulted_skill_ids=skill_ids,
                )
            )
        return candidates

    @staticmethod
    def _is_valid_python(source: str) -> bool:
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _inject_lru_cache(source: str, recursive_fns: tuple[str, ...]) -> str:
        lines = source.splitlines()
        out_lines: list[str] = []
        has_import = "lru_cache" in source
        if not has_import:
            out_lines.append("from functools import lru_cache")
            out_lines.append("")

        target_set = set(recursive_fns)
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if stripped.startswith("def "):
                fn_name = stripped[4:].split("(", 1)[0].strip()
                if fn_name in target_set:
                    out_lines.append(f"{indent}@lru_cache(maxsize=None)")
            out_lines.append(line)

        result = "\n".join(out_lines) + "\n"
        return result if HeuristicCodeMutator._is_valid_python(result) else source


class EvolutionaryLoop:
    """4-Step Self-Evolution Engine: Diff -> Mutate -> Verify (Rust Rayon) -> Select."""

    def __init__(
        self,
        skill_library: SkillLibrary,
        mutator: MutationGenerator | None = None,
        config: EvolutionConfig | None = None,
    ) -> None:
        self.skill_library = skill_library
        self.mutator: MutationGenerator = mutator if mutator is not None else HeuristicCodeMutator()
        self.config = config if config is not None else EvolutionConfig()
        self.sandbox_config = native.SandboxConfig(
            timeout_ms=self.config.timeout_ms,
            memory_limit_mb=self.config.memory_limit_mb,
        )

    def verify_candidates(
        self,
        candidates: list[ArchitectureCandidate],
        test_code: str,
        test_filename: str = "test_solution.py",
    ) -> list[native.VerificationResult]:
        """Step 3 (Verify): Materialize candidates in isolated workspaces and run
        parallel native Rust verification via `hybrid_ttt_native.verify_batch`."""
        with tempfile.TemporaryDirectory(prefix="hybrid_ttt_evo_") as root_tmp:
            root_path = Path(root_tmp)
            tasks: list[native.VerificationTask] = []

            for cand in candidates:
                ws = root_path / cand.candidate_id
                ws.mkdir(parents=True, exist_ok=True)
                (ws / cand.module_filename).write_text(cand.source_code, encoding="utf-8")
                (ws / test_filename).write_text(test_code, encoding="utf-8")

                compile_cmd = [sys.executable, "-m", "py_compile", cand.module_filename]
                test_cmd = [sys.executable, test_filename]
                tasks.append(
                    native.VerificationTask(
                        task_id=cand.candidate_id,
                        workdir=str(ws),
                        compile_cmd=compile_cmd,
                        lint_cmd=None,
                        test_cmd=test_cmd,
                        env_vars=[("PYTHONDONTWRITEBYTECODE", "1")],
                    )
                )

            return native.verify_batch(
                tasks,
                config=self.sandbox_config,
                num_threads=self.config.num_threads,
            )

    def run(
        self,
        initial_candidate: ArchitectureCandidate,
        test_code: str,
        problem_description: str = "",
    ) -> list[EvolutionStepReport]:
        """Execute full multi-generation Diff -> Mutate -> Verify -> Select evolution."""
        reports: list[EvolutionStepReport] = []
        baseline_code = initial_candidate.source_code
        current_best = initial_candidate

        # Establish initial parent score via Rust sandbox
        best_result = self.verify_candidates([current_best], test_code)[0]

        for gen in range(1, self.config.generations + 1):
            # 1. Diff
            diff_report = analyze_source_bottlenecks(baseline_code, current_best.source_code)

            # Query Graph Memory for relevant architectural principles
            query_text = f"{problem_description} {' '.join(diff_report.bottlenecks)}"
            retrieved = [r.skill for r in self.skill_library.query(query_text, top_k=3)]

            # 2. Mutate
            challengers = self.mutator.generate(
                parent=current_best,
                diff_report=diff_report,
                retrieved_skills=retrieved,
                max_mutations=self.config.mutations_per_generation,
            )

            # 3. Verify in parallel via Rust Rayon sandbox
            eval_pool = [current_best, *challengers]
            results = self.verify_candidates(eval_pool, test_code)

            # 4. Select Pareto-optimal candidate (passed > higher reward > lower duration_ms)
            paired = list(zip(eval_pool, results, strict=True))
            winner_cand, winner_res = max(
                paired,
                key=lambda item: (
                    1 if item[1].passed else 0,
                    round(item[1].reward, 6),
                    -int(item[1].duration_ms),
                ),
            )

            passed_count = sum(1 for r in results if r.passed)
            improved = winner_cand.candidate_id != current_best.candidate_id and (
                (winner_res.passed and not best_result.passed)
                or (winner_res.reward > best_result.reward + 1e-6)
                or (
                    winner_res.passed
                    and best_result.passed
                    and winner_res.duration_ms < best_result.duration_ms
                )
            )

            learned_skill_id: str | None = None
            if improved:
                learned_skill_id = f"evo_{winner_cand.mutation_name}"
                self.skill_library.upsert_skill(
                    skill_id=learned_skill_id,
                    name=winner_cand.mutation_name.replace("_", " ").title(),
                    category="optimization",
                    principle=(
                        f"Resolved bottlenecks {diff_report.bottlenecks} via "
                        f"{winner_cand.mutation_name} (reward={winner_res.reward:.3f}, "
                        f"latency={winner_res.duration_ms}ms)."
                    ),
                    code_pattern=winner_cand.mutation_name,
                    tags=diff_report.bottlenecks + (winner_cand.mutation_name,),
                    reward=winner_res.reward,
                    passed=winner_res.passed,
                )
                for consulted_id in winner_cand.consulted_skill_ids:
                    if consulted_id != learned_skill_id and self.skill_library.get(consulted_id):
                        self.skill_library.link_skills(
                            source_id=learned_skill_id,
                            target_id=consulted_id,
                            relation="refines",
                            weight=1.0,
                        )
                        self.skill_library.record_outcome(
                            consulted_id, reward=winner_res.reward, passed=winner_res.passed
                        )

                baseline_code = current_best.source_code
                current_best = winner_cand
                best_result = winner_res
            else:
                for skill in retrieved:
                    self.skill_library.record_outcome(
                        skill.skill_id, reward=winner_res.reward, passed=winner_res.passed
                    )

            reports.append(
                EvolutionStepReport(
                    generation=gen,
                    diff_report=diff_report,
                    evaluated_count=len(eval_pool),
                    passed_count=passed_count,
                    selected_candidate=current_best,
                    selected_result=best_result,
                    improved=improved,
                    learned_skill_id=learned_skill_id,
                )
            )

        return reports
