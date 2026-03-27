from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import UUID

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.drift_score import DriftScore
from app.models.health_report import HealthReport
from app.models.model_registry import ModelRegistry
from app.services import importance_scorer


def compute_overall_health_score(weighted_scores: list[float]) -> float:
    """Compute overall model health score from weighted drift scores.

    Args:
            weighted_scores: Weighted drift scores in [0, 1].

    Returns:
            Overall health score in [0, 100].
    """
    if not weighted_scores:
        return 100.0

    score = 100.0 * (1.0 - float(np.mean(weighted_scores)))
    return round(float(np.clip(score, 0.0, 100.0)), 3)


def compute_correlation(
    feature_drift: dict[str, float],
    output_drift: float | None,
) -> dict[str, float | int | None]:
    """Compute feature-to-output drift correlations.

    Args:
            feature_drift: Mapping of feature name to drift magnitude (PSI).
            output_drift: Output distribution shift metric, if available.

    Returns:
            Correlation summary with Pearson and Spearman values when available.
    """
    n_features = len(feature_drift)
    if output_drift is None or n_features < 3:
        return {"pearson_r": None, "spearman_r": None, "n_features": n_features}

    x = np.asarray(list(feature_drift.values()), dtype=float)
    y = np.full(shape=n_features, fill_value=float(output_drift), dtype=float)

    pearson_r: float | None = None
    spearman_r_value: float | None = None

    pearson_raw = pearsonr(x, y).statistic
    spearman_raw = spearmanr(x, y).statistic

    if np.isfinite(pearson_raw):
        pearson_r = float(pearson_raw)
    if np.isfinite(spearman_raw):
        spearman_r_value = float(spearman_raw)

    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r_value,
        "n_features": n_features,
    }


def _status_band(score: float) -> str:
    """Map health score to color band."""
    if score >= 80:
        return "GREEN"
    if score >= 60:
        return "YELLOW"
    return "RED"


def _render_markdown_report(
    model_name: str,
    week_start: date,
    week_end: date,
    overall_score: float,
    ranked_rows: list[dict],
    output_drift: float | None,
    correlation: dict[str, float | int | None],
    recommendations: list[str],
    red_count: int,
    yellow_count: int,
    green_count: int,
) -> str:
    """Render report markdown from prepared report components."""
    status = _status_band(overall_score)
    feature_count = len(ranked_rows)

    lines: list[str] = [
        "# DriftWatch Weekly Health Report",
        f"Model: {model_name}",
        f"Period: {week_start.isoformat()} to {week_end.isoformat()}",
        f"Overall Health Score: {overall_score}/100 ({status})",
        "",
        "## Executive Summary",
        (
            f"{feature_count} features monitored. {red_count} critical, "
            f"{yellow_count} moderate, {green_count} stable."
        ),
        (recommendations[0] if recommendations else "No immediate actions required."),
        "",
        "## Feature Drift Summary",
        "| Feature | PSI | KS p-value | JS Divergence | Weighted Score | Severity |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in ranked_rows:
        lines.append(
            "| {feature} | {psi:.3f} | {ks_p:.3f} | {js:.3f} | {weighted:.3f} | {severity} |".format(
                feature=row["feature_name"],
                psi=float(row.get("psi", 0.0)),
                ks_p=float(row.get("ks_pvalue", 0.0)),
                js=float(row.get("js_divergence", 0.0)),
                weighted=float(row.get("weighted_score", 0.0)),
                severity=str(row.get("severity", "green")),
            )
        )

    lines.extend(["", "## Output Distribution"])
    if output_drift is None:
        lines.append("No prediction data provided")
    else:
        lines.append(f"Output drift PSI: {output_drift:.3f}")

    lines.extend(["", "## Feature to Output Correlation"])
    lines.append(
        "Pearson r: {pearson}, Spearman r: {spearman}, features: {n}".format(
            pearson=(
                f"{float(correlation['pearson_r']):.3f}"
                if correlation["pearson_r"] is not None
                else "N/A"
            ),
            spearman=(
                f"{float(correlation['spearman_r']):.3f}"
                if correlation["spearman_r"] is not None
                else "N/A"
            ),
            n=int(correlation["n_features"]),
        )
    )

    lines.extend(["", "## Recommendations"])
    if recommendations:
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
    else:
        lines.append("- No recommendations")

    return "\n".join(lines)


async def generate_report(
    db: AsyncSession,
    model_id: UUID,
    week_start: date,
    week_end: date,
) -> HealthReport:
    """Generate and persist a weekly health report for one model.

    Args:
            db: Database session.
            model_id: Monitored model identifier.
            week_start: Inclusive week start date.
            week_end: Inclusive week end date.

    Returns:
            Persisted health report row.
    """
    model_result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.id == model_id)
    )
    model = model_result.scalar_one_or_none()
    model_name = model.name if model is not None else str(model_id)

    rows_result = await db.execute(
        select(DriftScore)
        .where(
            DriftScore.model_id == model_id,
            DriftScore.window_date >= week_start,
            DriftScore.window_date <= week_end,
        )
        .order_by(DriftScore.window_date.desc())
    )
    rows = list(rows_result.scalars().all())

    worst_by_feature: dict[str, DriftScore] = {}
    for row in rows:
        current = worst_by_feature.get(row.feature_name)
        current_psi = float(current.psi or 0.0) if current else -1.0
        row_psi = float(row.psi or 0.0)
        if row_psi > current_psi:
            worst_by_feature[row.feature_name] = row

    output_row = worst_by_feature.pop("__predictions__", None)
    output_drift = (
        float(output_row.psi) if output_row and output_row.psi is not None else None
    )

    feature_rows = list(worst_by_feature.values())
    feature_dicts = [
        {
            "feature_name": row.feature_name,
            "psi": float(row.psi or 0.0),
            "ks_pvalue": float(row.ks_pvalue or 0.0),
            "js_divergence": float(row.js_divergence or 0.0),
            "severity": row.severity or "green",
        }
        for row in feature_rows
    ]

    importances = await importance_scorer.load_importances(db=db, model_id=model_id)
    if not importances:
        importances = importance_scorer.equal_weight_importances(
            [row["feature_name"] for row in feature_dicts]
        )
    else:
        importances = importance_scorer.normalise_importances(importances)

    ranked_features = importance_scorer.rank_features_by_weighted_score(
        drift_scores=feature_dicts,
        importances=importances,
    )

    weighted_scores = [float(row["weighted_score"]) for row in ranked_features]
    overall_score = compute_overall_health_score(weighted_scores)

    feature_drift = {row["feature_name"]: float(row["psi"]) for row in ranked_features}
    correlation = compute_correlation(
        feature_drift=feature_drift, output_drift=output_drift
    )

    red_features = [row for row in ranked_features if row["severity"] == "red"]
    yellow_count = sum(1 for row in ranked_features if row["severity"] == "yellow")
    green_count = sum(1 for row in ranked_features if row["severity"] == "green")

    recommendations: list[str] = []
    if overall_score < 60:
        recommendations.append("Model retraining recommended")
    for row in red_features:
        recommendations.append(f"Investigate {row['feature_name']} immediately")
    if output_drift is not None and output_drift > 0.20:
        recommendations.append("Output distribution has significantly shifted")

    report_json = {
        "executive_summary": {
            "features_monitored": len(ranked_features),
            "red": len(red_features),
            "yellow": yellow_count,
            "green": green_count,
            "overall_score": overall_score,
        },
        "feature_drift_summary": ranked_features,
        "output_distribution": {
            "output_drift_psi": output_drift,
            "message": (
                "No prediction data provided"
                if output_drift is None
                else f"Output drift PSI={output_drift:.3f}"
            ),
        },
        "top_drifted_features": ranked_features[:5],
        "feature_to_output_correlation": correlation,
        "recommendations": recommendations,
    }

    report_markdown = _render_markdown_report(
        model_name=model_name,
        week_start=week_start,
        week_end=week_end,
        overall_score=overall_score,
        ranked_rows=ranked_features,
        output_drift=output_drift,
        correlation=correlation,
        recommendations=recommendations,
        red_count=len(red_features),
        yellow_count=yellow_count,
        green_count=green_count,
    )

    existing_result = await db.execute(
        select(HealthReport).where(
            HealthReport.model_id == model_id,
            HealthReport.week_start == week_start,
        )
    )
    existing = existing_result.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if existing is None:
        report = HealthReport(
            model_id=model_id,
            week_start=week_start,
            week_end=week_end,
            overall_score=overall_score,
            report_json=report_json,
            report_markdown=report_markdown,
            generated_at=now,
        )
        db.add(report)
    else:
        existing.week_end = week_end
        existing.overall_score = overall_score
        existing.report_json = report_json
        existing.report_markdown = report_markdown
        existing.generated_at = now
        report = existing

    await db.commit()
    await db.refresh(report)
    return report


async def list_reports_for_model(
    db: AsyncSession, model_id: UUID
) -> list[HealthReport]:
    """List health reports for one model in newest-first order."""
    result = await db.execute(
        select(HealthReport)
        .where(HealthReport.model_id == model_id)
        .order_by(HealthReport.week_start.desc())
    )
    return list(result.scalars().all())


async def get_latest_report_for_model(
    db: AsyncSession, model_id: UUID
) -> HealthReport | None:
    """Fetch latest health report for one model."""
    result = await db.execute(
        select(HealthReport)
        .where(HealthReport.model_id == model_id)
        .order_by(HealthReport.week_start.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_report_for_week(
    db: AsyncSession,
    model_id: UUID,
    week_start: date,
) -> HealthReport | None:
    """Fetch a specific weekly report by model and week_start."""
    result = await db.execute(
        select(HealthReport).where(
            HealthReport.model_id == model_id,
            HealthReport.week_start == week_start,
        )
    )
    return result.scalar_one_or_none()
