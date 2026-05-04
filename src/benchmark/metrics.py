from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import save_csv


FAMILY_ALIASES = {
    "direct": "direct",
    "direct_target": "direct",
    "target": "direct",
    "indirect": "indirect",
    "indirect_strict": "indirect",
    "indirect_broad": "indirect",
    "neighbor": "neighbor",
    "neighbors": "neighbor",
    "control": "control",
    "controls": "control",
}


def _normalize_text(value: object) -> str:
    return str(value).strip().lower()


def _normalize_prompt_family(value: object) -> str:
    key = _normalize_text(value)
    return FAMILY_ALIASES.get(key, key)


def _safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def _acc(df: pd.DataFrame) -> float:
    if len(df) == 0 or "is_correct" not in df.columns:
        return float("nan")
    valid = df["is_correct"].dropna()
    if len(valid) == 0:
        return float("nan")
    return float(valid.mean())


def _to_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert mixed prediction flags to pandas nullable boolean.
    Supports bools, 0/1, and common string literals; unknown values -> <NA>.
    """
    if series.dtype == bool or str(series.dtype) == "boolean":
        return series.astype("boolean")

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    normalized = series.map(lambda x: str(x).strip().lower() if pd.notna(x) else x)
    mapped = normalized.map(lambda x: mapping.get(x, pd.NA) if pd.notna(x) else pd.NA)
    return mapped.astype("boolean")


def _prepare_prediction_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    for col in ["target_concept", "intended_label", "model_name", "domain"]:
        if col in df.columns:
            df[col] = df[col].map(_normalize_text)
    if "prompt_family" in df.columns:
        df["prompt_family_raw"] = df["prompt_family"].astype(str)
        df["prompt_family"] = df["prompt_family"].map(_normalize_prompt_family)
    if "is_target_pred" in df.columns:
        df["is_target_pred"] = _to_bool_series(df["is_target_pred"])
    return df


def _normalize_concept_map(raw_map: dict[str, list[str]] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, values in (raw_map or {}).items():
        out[_normalize_text(key)] = [_normalize_text(v) for v in values]
    return out


def _metric_reason(valid_count: int, base_present: bool, map_present: bool, kind: str) -> str:
    if valid_count > 0:
        return "computed"
    if not base_present:
        return "base_model_missing_for_target"
    if not map_present:
        return f"{kind}_map_missing_for_target"
    return f"no_valid_{kind}_concepts_after_filtering"


def compute_per_target_metrics(
    pred_df: pd.DataFrame,
    base_model_name: str,
    compute_cp: bool = True,
    neighbor_map: dict[str, list[str]] | None = None,
    control_map: dict[str, list[str]] | None = None,
    min_base_acc_for_normalization: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - per_target_metrics
      - detailed_breakdowns (per concept in NP/CP)
      - debug_metric_inputs (per target/model diagnostics)
    """
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    if len(pred_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = _prepare_prediction_df(pred_df)
    base_model_name = _normalize_text(base_model_name)
    neighbor_map = _normalize_concept_map(neighbor_map)
    control_map = _normalize_concept_map(control_map)

    targets = sorted(df["target_concept"].dropna().astype(str).unique().tolist())
    models = sorted(df["model_name"].dropna().astype(str).unique().tolist())
    domain_cardinality = int(df["domain"].dropna().astype(str).nunique()) if "domain" in df.columns else 0

    for target in targets:
        target_df = df[df["target_concept"] == target]
        if len(target_df) == 0:
            continue
        domain_mode = target_df["domain"].mode() if "domain" in target_df.columns else pd.Series(dtype=str)
        target_domain = str(domain_mode.iloc[0]) if len(domain_mode) else ""
        base_target = target_df[target_df["model_name"] == base_model_name]
        base_present = len(base_target) > 0

        for model in models:
            md = target_df[target_df["model_name"] == model]
            if len(md) == 0:
                continue

            direct_df = md[md["prompt_family"] == "direct"]
            indirect_df = md[md["prompt_family"] == "indirect"]
            neighbor_df = md[md["prompt_family"] == "neighbor"]
            control_df = md[md["prompt_family"] == "control"]

            if len(direct_df) > 0 and "is_target_pred" in direct_df.columns:
                ua = _safe_mean((~direct_df["is_target_pred"]).astype("Float64"))
            else:
                ua = float("nan")
            if len(indirect_df) > 0 and "is_target_pred" in indirect_df.columns:
                irr = _safe_mean(indirect_df["is_target_pred"].astype("Float64"))
            else:
                irr = float("nan")

            has_domain_metadata = "domain" in md.columns and md["domain"].notna().any()
            if has_domain_metadata and domain_cardinality > 1:
                ira_df = md[(md["intended_label"] != target) & (md["domain"] == target_domain)]
                ira = _acc(ira_df)
                cra_df = md[md["domain"] != target_domain]
                cra = _acc(cra_df) if len(cra_df) > 0 else float("nan")
                cra_reason = "computed" if len(cra_df) > 0 else "no_other_domain_rows"
            elif has_domain_metadata:
                ira_df = md[(md["intended_label"] != target) & (md["domain"] == target_domain)]
                ira = _acc(ira_df)
                cra = float("nan")
                cra_reason = "single_domain_dataset"
            else:
                ira = float("nan")
                cra = float("nan")
                cra_reason = "domain_metadata_missing"

            np_score = float("nan")
            cp_score = float("nan")
            damage_gap = float("nan")

            neighbor_map_present = target in neighbor_map
            control_map_present = target in control_map
            neighbor_allowed = set(neighbor_map.get(target, []))
            control_allowed = set(control_map.get(target, []))

            neighbor_valid_count = 0
            control_valid_count = 0
            neighbor_filtered_by_map = 0
            control_filtered_by_map = 0

            ratios_np: list[float] = []
            neighbor_concepts = sorted(neighbor_df["intended_label"].dropna().astype(str).unique().tolist())
            for concept in neighbor_concepts:
                if neighbor_map_present and concept not in neighbor_allowed:
                    neighbor_filtered_by_map += 1
                    detail_rows.append(
                        {
                            "target_concept": target,
                            "model_name": model,
                            "metric_component": "NP",
                            "prompt_family": "neighbor",
                            "concept": concept,
                            "unlearned_n": int(len(neighbor_df[neighbor_df["intended_label"] == concept])),
                            "base_n": 0,
                            "unlearned_acc": _acc(neighbor_df[neighbor_df["intended_label"] == concept]),
                            "base_acc": float("nan"),
                            "ratio": float("nan"),
                            "normalization_status": "filtered_by_neighbor_map",
                        }
                    )
                    continue

                u_df = neighbor_df[neighbor_df["intended_label"] == concept]
                b_df = base_target[
                    (base_target["prompt_family"] == "neighbor") & (base_target["intended_label"] == concept)
                ]
                u_acc = _acc(u_df)
                b_acc = _acc(b_df)
                ratio = float("nan")
                if not base_present:
                    normalization_status = "base_model_missing_for_target"
                elif len(b_df) == 0:
                    normalization_status = "base_neighbor_rows_missing"
                elif pd.isna(b_acc):
                    normalization_status = "base_acc_nan"
                elif b_acc <= min_base_acc_for_normalization:
                    normalization_status = "base_acc_below_threshold"
                elif pd.isna(u_acc):
                    normalization_status = "unlearned_acc_nan"
                else:
                    normalization_status = "ok"
                    ratio = float(u_acc / b_acc)
                    ratios_np.append(ratio)
                    neighbor_valid_count += 1
                detail_rows.append(
                    {
                        "target_concept": target,
                        "model_name": model,
                        "metric_component": "NP",
                        "prompt_family": "neighbor",
                        "concept": concept,
                        "unlearned_n": int(len(u_df)),
                        "base_n": int(len(b_df)),
                        "unlearned_acc": u_acc,
                        "base_acc": b_acc,
                        "ratio": ratio,
                        "normalization_status": normalization_status,
                    }
                )
            if ratios_np:
                np_score = float(np.mean(ratios_np))

            ratios_cp: list[float] = []
            if compute_cp:
                control_concepts = sorted(control_df["intended_label"].dropna().astype(str).unique().tolist())
                for concept in control_concepts:
                    if control_map_present and concept not in control_allowed:
                        control_filtered_by_map += 1
                        detail_rows.append(
                            {
                                "target_concept": target,
                                "model_name": model,
                                "metric_component": "CP",
                                "prompt_family": "control",
                                "concept": concept,
                                "unlearned_n": int(len(control_df[control_df["intended_label"] == concept])),
                                "base_n": 0,
                                "unlearned_acc": _acc(control_df[control_df["intended_label"] == concept]),
                                "base_acc": float("nan"),
                                "ratio": float("nan"),
                                "normalization_status": "filtered_by_control_map",
                            }
                        )
                        continue

                    u_df = control_df[control_df["intended_label"] == concept]
                    b_df = base_target[
                        (base_target["prompt_family"] == "control") & (base_target["intended_label"] == concept)
                    ]
                    u_acc = _acc(u_df)
                    b_acc = _acc(b_df)
                    ratio = float("nan")
                    if not base_present:
                        normalization_status = "base_model_missing_for_target"
                    elif len(b_df) == 0:
                        normalization_status = "base_control_rows_missing"
                    elif pd.isna(b_acc):
                        normalization_status = "base_acc_nan"
                    elif b_acc <= min_base_acc_for_normalization:
                        normalization_status = "base_acc_below_threshold"
                    elif pd.isna(u_acc):
                        normalization_status = "unlearned_acc_nan"
                    else:
                        normalization_status = "ok"
                        ratio = float(u_acc / b_acc)
                        ratios_cp.append(ratio)
                        control_valid_count += 1
                    detail_rows.append(
                        {
                            "target_concept": target,
                            "model_name": model,
                            "metric_component": "CP",
                            "prompt_family": "control",
                            "concept": concept,
                            "unlearned_n": int(len(u_df)),
                            "base_n": int(len(b_df)),
                            "unlearned_acc": u_acc,
                            "base_acc": b_acc,
                            "ratio": ratio,
                            "normalization_status": normalization_status,
                        }
                    )
                if ratios_cp:
                    cp_score = float(np.mean(ratios_cp))

            if pd.notna(cp_score) and pd.notna(np_score):
                damage_gap = float(cp_score - np_score)

            rows.append(
                {
                    "target_concept": target,
                    "model_name": model,
                    "UA": ua,
                    "IRR": irr,
                    "IRA": ira,
                    "CRA": cra,
                    "NP": np_score,
                    "CP": cp_score,
                    "DamageGap": damage_gap,
                    "n_direct": int(len(direct_df)),
                    "n_indirect": int(len(indirect_df)),
                    "n_neighbor": int(len(neighbor_df)),
                    "n_control": int(len(control_df)),
                }
            )

            debug_rows.append(
                {
                    "target_concept": target,
                    "model_name": model,
                    "base_model_name": base_model_name,
                    "base_model_present_for_target": bool(base_present),
                    "n_base_rows_for_target": int(len(base_target)),
                    "n_direct_rows": int(len(direct_df)),
                    "n_indirect_rows": int(len(indirect_df)),
                    "n_neighbor_rows": int(len(neighbor_df)),
                    "n_control_rows": int(len(control_df)),
                    "has_domain_metadata": bool(has_domain_metadata),
                    "n_domains_total": int(domain_cardinality),
                    "target_domain": target_domain if target_domain else "NA",
                    "CRA_reason": cra_reason,
                    "neighbor_map_present": bool(neighbor_map_present),
                    "neighbor_map_size": int(len(neighbor_map.get(target, []))),
                    "control_map_present": bool(control_map_present),
                    "control_map_size": int(len(control_map.get(target, []))),
                    "neighbor_concepts_seen": int(len(neighbor_concepts)),
                    "control_concepts_seen": int(control_df["intended_label"].dropna().astype(str).nunique()),
                    "neighbor_concepts_filtered_by_map": int(neighbor_filtered_by_map),
                    "control_concepts_filtered_by_map": int(control_filtered_by_map),
                    "neighbor_valid_concepts_for_np": int(neighbor_valid_count),
                    "control_valid_concepts_for_cp": int(control_valid_count),
                    "NP_reason": _metric_reason(neighbor_valid_count, base_present, neighbor_map_present, "neighbor"),
                    "CP_reason": _metric_reason(control_valid_count, base_present, control_map_present, "control")
                    if compute_cp
                    else "cp_disabled",
                    "rows_dropped_due_to_na_or_filtering": int(
                        max(0, len(neighbor_concepts) - neighbor_valid_count)
                        + max(0, int(control_df["intended_label"].dropna().astype(str).nunique()) - control_valid_count)
                    ),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(detail_rows), pd.DataFrame(debug_rows)


def aggregate_metrics(per_target_metrics: pd.DataFrame) -> pd.DataFrame:
    if len(per_target_metrics) == 0:
        return pd.DataFrame()
    metric_cols = ["UA", "IRR", "IRA", "CRA", "NP", "CP", "DamageGap"]
    rows = []
    for model, group in per_target_metrics.groupby("model_name"):
        row = {"model_name": model, "n_targets": int(group["target_concept"].nunique())}
        for m in metric_cols:
            row[m] = float(group[m].mean(skipna=True)) if m in group.columns else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def save_metrics_outputs(
    per_target_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    eval_dir: str,
    prefix: str = "",
    debug_df: pd.DataFrame | None = None,
) -> None:
    stem = f"{prefix}_" if prefix else ""
    save_csv(f"{eval_dir}/{stem}per_target_metrics.csv", per_target_df, na_rep="NA")
    save_csv(f"{eval_dir}/{stem}aggregated_metrics.csv", aggregated_df, na_rep="NA")
    save_csv(f"{eval_dir}/{stem}detailed_breakdowns.csv", detailed_df, na_rep="NA")
    if len(detailed_df) > 0 and "metric_component" in detailed_df.columns:
        neighbor_df = detailed_df[detailed_df["metric_component"] == "NP"].copy()
        control_df = detailed_df[detailed_df["metric_component"] == "CP"].copy()
        save_csv(f"{eval_dir}/{stem}per_neighbor_metrics.csv", neighbor_df, na_rep="NA")
        save_csv(f"{eval_dir}/{stem}per_control_metrics.csv", control_df, na_rep="NA")
    if debug_df is not None:
        save_csv(f"{eval_dir}/{stem}debug_metric_inputs.csv", debug_df, na_rep="NA")
