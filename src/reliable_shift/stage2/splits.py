from __future__ import annotations

import hashlib
import pandas as pd


ROLES = ("source_train", "source_tune", "source_probability_calibration", "source_conformal_calibration", "source_id_test")


def entity_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"stage2|{seed}|{value}".encode()).hexdigest()


def assign_roles(frame: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    result = frame.loc[:, ["entity_id", "is_target_domain"]].copy()
    result["entity_hash"] = [entity_hash(str(value), split_seed) for value in result["entity_id"]]
    source = ~result["is_target_domain"].astype(bool)
    source_bucket = result.loc[source, "entity_hash"].str[:8].apply(lambda value: int(value, 16) % 100)
    source_roles = pd.cut(source_bucket, [-1, 59, 69, 79, 89, 99], labels=ROLES)
    result.loc[source, "role"] = source_roles.astype("string")
    target_bucket = result.loc[~source, "entity_hash"].str[:8].apply(lambda value: int(value, 16) % 100)
    result.loc[~source, "role"] = target_bucket.map(lambda value: "target_adaptation_pool" if value < 40 else "target_final_test")
    if result.duplicated("entity_id").any():
        raise ValueError("entities are not unique before role assignment")
    return result.drop(columns=["entity_id", "is_target_domain"])


def capped_role_frames(
    frame: pd.DataFrame, split_seed: int, role_caps: dict[str, int | None]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Apply seven deterministic roles and label-blind per-role entity caps."""
    assigned = assign_roles(frame, split_seed)
    work = frame.copy()
    work["entity_hash"] = assigned["entity_hash"].to_numpy()
    work["role"] = assigned["role"].to_numpy()
    roles: dict[str, pd.DataFrame] = {}
    audit_rows = []
    domain_prevalence = work.groupby(work["is_target_domain"].astype(bool))["target"].mean()
    for role in (
        "source_train", "source_tune", "source_probability_calibration",
        "source_conformal_calibration", "source_id_test",
        "target_adaptation_pool", "target_final_test",
    ):
        subset = work.loc[work["role"].eq(role)].sort_values("entity_hash", kind="stable")
        cap = role_caps.get(role)
        if cap is not None:
            subset = subset.iloc[: int(cap)]
        subset = subset.copy()
        roles[role] = subset
        positive = int(subset["target"].sum())
        prevalence = float(subset["target"].mean()) if len(subset) else float("nan")
        target_domain = role.startswith("target_")
        audit_rows.append({
            "split_seed": int(split_seed),
            "role": role,
            "rows": int(len(subset)),
            "entities": int(subset["entity_id"].nunique()),
            "positive_count": positive,
            "negative_count": int(len(subset) - positive),
            "prevalence": prevalence,
            "difference_from_domain_prevalence": prevalence - float(domain_prevalence.loc[target_domain]),
            "role_hash": _role_hash(subset["entity_hash"]),
            "cap_reads_labels": False,
        })
    return roles, pd.DataFrame(audit_rows)


def _role_hash(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()
