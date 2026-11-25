from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

EdgeRelation = Literal["refines", "composes_with", "supersedes", "mitigates"]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) >= 2}


@dataclass(slots=True)
class SkillNode:
    """Verified architectural principle stored in Graph Memory."""

    skill_id: str
    name: str
    category: str
    principle: str
    code_pattern: str
    tags: tuple[str, ...]
    success_count: int = 1
    failure_count: int = 0
    avg_reward: float = 0.0
    created_step: int = 0
    updated_step: int = 0

    @property
    def success_rate(self) -> float:
        # Laplace-smoothed empirical success rate
        return (self.success_count + 1.0) / (self.success_count + self.failure_count + 2.0)

    def searchable_tokens(self) -> set[str]:
        parts = [self.name, self.category, self.principle, self.code_pattern, " ".join(self.tags)]
        return _tokenize(" ".join(parts))


@dataclass(frozen=True, slots=True)
class SkillEdge:
    """Directed relationship between two architectural skill nodes."""

    source_id: str
    target_id: str
    relation: EdgeRelation
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class SkillQueryResult:
    skill: SkillNode
    score: float
    hop_distance: int


class SkillLibrary:
    """Graph Memory storing, querying, and reinforcing verified coding principles."""

    def __init__(self) -> None:
        self._nodes: dict[str, SkillNode] = {}
        self._outgoing: dict[str, list[SkillEdge]] = {}
        self._incoming: dict[str, list[SkillEdge]] = {}
        self._step_counter: int = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, skill_id: str) -> SkillNode | None:
        return self._nodes.get(skill_id)

    def list_nodes(self) -> list[SkillNode]:
        return list(self._nodes.values())

    def list_edges(self) -> list[SkillEdge]:
        edges: list[SkillEdge] = []
        for edge_list in self._outgoing.values():
            edges.extend(edge_list)
        return edges

    def upsert_skill(
        self,
        skill_id: str,
        name: str,
        category: str,
        principle: str,
        code_pattern: str,
        tags: tuple[str, ...] = (),
        reward: float = 0.0,
        passed: bool = True,
    ) -> SkillNode:
        self._step_counter += 1
        existing = self._nodes.get(skill_id)
        if existing is None:
            node = SkillNode(
                skill_id=skill_id,
                name=name,
                category=category,
                principle=principle,
                code_pattern=code_pattern,
                tags=tuple(sorted(set(tags))),
                success_count=1 if passed else 0,
                failure_count=0 if passed else 1,
                avg_reward=reward,
                created_step=self._step_counter,
                updated_step=self._step_counter,
            )
            self._nodes[skill_id] = node
            self._outgoing.setdefault(skill_id, [])
            self._incoming.setdefault(skill_id, [])
            return node

        total_trials = existing.success_count + existing.failure_count
        existing.name = name
        existing.category = category
        existing.principle = principle
        existing.code_pattern = code_pattern
        existing.tags = tuple(sorted(set(existing.tags) | set(tags)))
        if passed:
            existing.success_count += 1
        else:
            existing.failure_count += 1
        existing.avg_reward = (existing.avg_reward * total_trials + reward) / float(
            total_trials + 1
        )
        existing.updated_step = self._step_counter
        return existing

    def link_skills(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation,
        weight: float = 1.0,
    ) -> SkillEdge:
        if source_id == target_id:
            raise ValueError("self-loop edges are not permitted in SkillLibrary")
        if source_id not in self._nodes or target_id not in self._nodes:
            raise KeyError(f"both source ({source_id}) and target ({target_id}) must exist")
        if weight <= 0.0 or not math.isfinite(weight):
            raise ValueError("edge weight must be finite and positive")

        edge = SkillEdge(source_id=source_id, target_id=target_id, relation=relation, weight=weight)
        out_edges = self._outgoing[source_id]
        for idx, existing in enumerate(out_edges):
            if existing.target_id == target_id and existing.relation == relation:
                out_edges[idx] = edge
                break
        else:
            out_edges.append(edge)

        in_edges = self._incoming[target_id]
        for idx, existing in enumerate(in_edges):
            if existing.source_id == source_id and existing.relation == relation:
                in_edges[idx] = edge
                break
        else:
            in_edges.append(edge)

        return edge

    def record_outcome(self, skill_id: str, reward: float, passed: bool) -> None:
        node = self._nodes.get(skill_id)
        if node is None:
            raise KeyError(f"unknown skill_id: {skill_id}")
        self._step_counter += 1
        trials = node.success_count + node.failure_count
        if passed:
            node.success_count += 1
        else:
            node.failure_count += 1
        node.avg_reward = (node.avg_reward * trials + reward) / float(trials + 1)
        node.updated_step = self._step_counter

    def query(
        self,
        query_text: str,
        category: str | None = None,
        top_k: int = 5,
        hops: int = 2,
        damping: float = 0.55,
    ) -> list[SkillQueryResult]:
        """Hybrid lexical + multi-hop graph-propagated retrieval weighted by
        empirical verification reward and Laplace-smoothed success rate."""
        if not self._nodes or top_k <= 0:
            return []

        q_tokens = _tokenize(query_text)
        num_docs = len(self._nodes)

        # Compute IDF across stored nodes
        doc_freq: dict[str, int] = {}
        node_tokens: dict[str, set[str]] = {}
        for sid, node in self._nodes.items():
            toks = node.searchable_tokens()
            node_tokens[sid] = toks
            for tok in toks:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        base_scores: dict[str, float] = {}
        hop_map: dict[str, int] = {}
        for sid, node in self._nodes.items():
            if category is not None and node.category != category:
                base_scores[sid] = 0.0
                continue
            overlap = q_tokens & node_tokens[sid]
            lexical = sum(
                math.log((num_docs + 1.0) / (doc_freq.get(t, 0) + 1.0)) + 1.0 for t in overlap
            )
            utility_multiplier = 0.5 + 0.5 * node.success_rate + max(0.0, node.avg_reward)
            score = lexical * utility_multiplier
            base_scores[sid] = score
            if score > 0.0:
                hop_map[sid] = 0

        # Multi-hop propagation along outgoing and incoming edges
        scores = dict(base_scores)
        frontier = {sid: sc for sid, sc in base_scores.items() if sc > 0.0}
        for hop in range(1, max(0, hops) + 1):
            next_frontier: dict[str, float] = {}
            decay = damping**hop
            for sid, seed_score in frontier.items():
                neighbors = [
                    (e.target_id, e.weight) for e in self._outgoing.get(sid, ())
                ] + [(e.source_id, e.weight * 0.7) for e in self._incoming.get(sid, ())]

                for nb_id, edge_w in neighbors:
                    nb_node = self._nodes[nb_id]
                    if category is not None and nb_node.category != category:
                        continue
                    propagated = seed_score * decay * min(edge_w, 2.0) * nb_node.success_rate
                    if propagated > 0.0:
                        scores[nb_id] = scores.get(nb_id, 0.0) + propagated
                        next_frontier[nb_id] = next_frontier.get(nb_id, 0.0) + propagated
                        hop_map.setdefault(nb_id, hop)
            frontier = next_frontier
            if not frontier:
                break

        ranked = sorted(
            (
                SkillQueryResult(
                    skill=self._nodes[sid],
                    score=sc,
                    hop_distance=hop_map.get(sid, hops + 1),
                )
                for sid, sc in scores.items()
                if sc > 0.0
            ),
            key=lambda r: (r.score, r.skill.avg_reward, r.skill.success_rate),
            reverse=True,
        )
        return ranked[:top_k]

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step_counter": self._step_counter,
            "nodes": [asdict(n) for n in self._nodes.values()],
            "edges": [asdict(e) for e in self.list_edges()],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SkillLibrary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lib = cls()
        lib._step_counter = int(data.get("step_counter", 0))
        for raw_node in data.get("nodes", []):
            node = SkillNode(
                skill_id=str(raw_node["skill_id"]),
                name=str(raw_node["name"]),
                category=str(raw_node["category"]),
                principle=str(raw_node["principle"]),
                code_pattern=str(raw_node["code_pattern"]),
                tags=tuple(str(t) for t in raw_node.get("tags", [])),
                success_count=int(raw_node.get("success_count", 1)),
                failure_count=int(raw_node.get("failure_count", 0)),
                avg_reward=float(raw_node.get("avg_reward", 0.0)),
                created_step=int(raw_node.get("created_step", 0)),
                updated_step=int(raw_node.get("updated_step", 0)),
            )
            lib._nodes[node.skill_id] = node
            lib._outgoing.setdefault(node.skill_id, [])
            lib._incoming.setdefault(node.skill_id, [])

        for raw_edge in data.get("edges", []):
            lib.link_skills(
                source_id=str(raw_edge["source_id"]),
                target_id=str(raw_edge["target_id"]),
                relation=raw_edge["relation"],
                weight=float(raw_edge.get("weight", 1.0)),
            )
        return lib
