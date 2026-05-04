"""
graph_utils.py

Dataset-agnostic anchor construction and pi computation.

Supports:
- Edge-list graphs (e.g., UnlearnCanvas style_graph_25.json) with relation="parent_of"
- Superclass mapping graphs (e.g., CIFAR-100 superclasses.json) where parent -> children

Anchor definition (default):
- parent: unique parent of a child (if exists)
- peers: siblings under the same parent (excluding the child)
- anchor set A(u): {parent} U peers
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Literal, Union
from collections import defaultdict

import json
import networkx as nx
import torch


PiMode = Literal["fixed", "embedding_softmax"]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=0)


def resolve_style_id(x: str) -> str:
    # Accept "Pencil Drawing" or "style_Pencil_Drawing"
    return x if x.startswith("style_") else "style_" + x.replace(" ", "_")

def resolve_cifar_fine_id(x: str) -> str:
    # Accept "beaver" or "fine_beaver"
    return x if x.startswith("fine_") else "fine_" + x

def resolve_cifar_coarse_id(x: str) -> str:
    # Accept "aquatic mammals" or "coarse_aquatic mammals"
    return x if x.startswith("coarse_") else "coarse_" + x


@dataclass
class GraphConfig:
    # relation label for hierarchy edges
    hierarchy_relation: str = "parent_of"
    # Optional root node name to exclude (if you ever add one)
    root_names: Tuple[str, ...] = ("root",)
    # If True, enforce each child has <= 1 parent for hierarchy edges
    enforce_single_parent: bool = True


class AnchorGraph:
    def __init__(self, G: nx.DiGraph, cfg: GraphConfig = GraphConfig()):
        self.G = G
        self.cfg = cfg

    # ---------- hierarchy helpers ----------
    def parents(self, u: str) -> List[str]:
        out = []
        for v in self.G.predecessors(u):
            rel = self.G.edges.get((v, u), {}).get("relation")
            if rel == self.cfg.hierarchy_relation:
                out.append(v)
        return out

    def children(self, u: str) -> List[str]:
        out = []
        for v in self.G.successors(u):
            rel = self.G.edges.get((u, v), {}).get("relation")
            if rel == self.cfg.hierarchy_relation:
                out.append(v)
        return out

    def is_root(self, u: str) -> bool:
        if u in self.cfg.root_names:
            return True
        return self.G.nodes.get(u, {}).get("type") == "root"

    def check_single_parent(self) -> None:
        if not self.cfg.enforce_single_parent:
            return
        for n in self.G.nodes:
            ps = self.parents(n)
            if len(ps) > 1:
                raise ValueError(f"Node {n} has multiple parents under relation={self.cfg.hierarchy_relation}: {ps}")

    # ---------- anchor construction ----------
    def parent_and_peers(self, u: str) -> Tuple[Optional[str], List[str]]:
        ps = self.parents(u)
        if len(ps) == 0:
            return None, []
        if self.cfg.enforce_single_parent and len(ps) > 1:
            raise ValueError(f"{u} has multiple parents: {ps}")
        parent = ps[0]

        sibs = [c for c in self.children(parent) if c != u]
        return parent, sibs

    def anchor_set(self, u: str) -> List[str]:
        parent, peers = self.parent_and_peers(u)
        anchors = []
        if parent is not None and (not self.is_root(parent)):
            anchors.append(parent)
        anchors.extend([p for p in peers if not self.is_root(p)])
        # de-dup keep order
        seen = set()
        out = []
        for a in anchors:
            if a not in seen:
                out.append(a); seen.add(a)
        return out

    # ---------- pi computation ----------
    def compute_pi(
        self,
        u: str,
        anchors: List[str],
        mode: PiMode = "fixed",
        *,
        alpha: float = 0.5,
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Return pi(a | u) over anchors.
        - mode="fixed": parent gets alpha, remaining mass split uniformly over peers (if any)
        - mode="embedding_softmax": cosine-softmax between embedding[u] and embedding[a]
        """
        if len(anchors) == 0:
            return {}

        if mode == "fixed":
            parent, peers = self.parent_and_peers(u)
            # If parent isn't in anchors (e.g. no parent), just uniform
            if parent is None or parent not in anchors:
                p = 1.0 / len(anchors)
                return {a: p for a in anchors}

            # peers are anchors excluding parent
            peer_anchors = [a for a in anchors if a != parent]
            if len(peer_anchors) == 0:
                return {parent: 1.0}

            rem = 1.0 - alpha
            per_peer = rem / len(peer_anchors)
            pi = {parent: float(alpha)}
            for a in peer_anchors:
                pi[a] = float(per_peer)
            return pi

        if mode == "embedding_softmax":
            if embeddings is None:
                raise ValueError("embeddings must be provided for mode='embedding_softmax'")
            if u not in embeddings:
                raise KeyError(f"Missing embedding for u={u}")
            zu = embeddings[u]

            scores = []
            for a in anchors:
                if a not in embeddings:
                    raise KeyError(f"Missing embedding for anchor={a}")
                za = embeddings[a]
                sim = _cosine(zu, za) / temperature
                scores.append(sim)

            probs = torch.softmax(torch.stack(scores), dim=0)
            return {a: float(probs[i].item()) for i, a in enumerate(anchors)}

        raise ValueError(f"Unknown mode: {mode}")


# ----------------- loaders -----------------

def load_edge_list_graph(path: str, cfg: GraphConfig = GraphConfig()) -> AnchorGraph:
    """
    Load graphs like style_graph_25.json:
      { "nodes": [{"id":..}, ...], "edges": [{"src":..,"dst":..,"relation":"parent_of"}, ...] }
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    G = nx.DiGraph()
    for n in data.get("nodes", []):
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in data.get("edges", []):
        G.add_edge(e["src"], e["dst"], relation=e.get("relation", cfg.hierarchy_relation))
    ag = AnchorGraph(G, cfg)
    ag.check_single_parent()
    return ag


def load_superclass_graph(superclasses_path: str, cfg: GraphConfig = GraphConfig(hierarchy_relation="parent_of")) -> AnchorGraph:
    """
    Load CIFAR-100 style mapping like superclasses.json:
      { "fish": {"children": ["shark", ...]}, ... }
    Creates nodes:
      coarse_fish (parent), fine_shark (child), etc.
    """
    data = json.load(open(superclasses_path, "r", encoding="utf-8"))
    G = nx.DiGraph()

    for parent_name, info in data.items():
        parent_id = f"coarse_{parent_name}"
        G.add_node(parent_id, name=parent_name, type="coarse")

        for child in info["children"]:
            child_id = f"fine_{child}"
            if child_id not in G:
                G.add_node(child_id, name=child, type="fine")
            G.add_edge(parent_id, child_id, relation=cfg.hierarchy_relation)

    ag = AnchorGraph(G, cfg)
    ag.check_single_parent()
    return ag


def cifar_node_id_fine(name: str) -> str:
    return f"fine_{name}"

def cifar_node_id_coarse(name: str) -> str:
    return f"coarse_{name}"
