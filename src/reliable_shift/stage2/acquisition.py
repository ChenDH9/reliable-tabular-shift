from __future__ import annotations

import hashlib
import numpy as np

from .splits import entity_hash


def label_blind_order(entity_ids, seed: int) -> np.ndarray:
    """Order target entities without accepting or reading labels."""
    keys = [hashlib.sha256(f"{seed}|{value}".encode()).hexdigest() for value in entity_ids]
    return np.argsort(keys, kind="stable")


def nested_budget_indices(entity_ids, budgets: list[int], seed: int) -> dict[int, np.ndarray]:
    order = label_blind_order(entity_ids, seed)
    return {int(budget): order[: min(int(budget), len(order))] for budget in sorted(budgets)}


def _digest(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def canonical_blind_budget_samples(
    *,
    dataset: str,
    split_seed: int,
    adaptation_seed: int,
    entity_ids,
    budgets: list[int],
) -> dict[int, dict[str, object]]:
    """Canonical label-blind nested samples used by manifests and the formal runner.

    `dataset` is deliberately part of the signature/audit record but not the ordering:
    entity IDs are already task scoped. No labels or model outputs are accepted.
    """
    del dataset
    hashes = np.asarray([entity_hash(str(value), split_seed) for value in entity_ids], dtype=str)
    canonical = np.sort(hashes)
    ordered = canonical[np.random.default_rng(adaptation_seed).permutation(len(canonical))]
    pool_hash = _digest(canonical)
    result: dict[int, dict[str, object]] = {}
    for budget in sorted(map(int, budgets)):
        selected = ordered[: min(budget, len(ordered))]
        result[budget] = {
            "selected_count": int(len(selected)),
            "selected_entity_hashes": selected,
            "sample_hash": _digest(selected),
            "pool_hash": pool_hash,
            "available": len(ordered) >= budget,
        }
    return result
