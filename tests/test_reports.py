from datetime import date, datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.schemas.report import HealthReportResponse

from app.services.report_generator import _render_markdown_report


def test_markdown_reports_statistic_without_inventing_pvalue() -> None:
    markdown = _render_markdown_report(
        model_name="test-model",
        week_start=date(2026, 3, 16),
        week_end=date(2026, 3, 22),
        overall_score=83.33333333333334,
        ranked_rows=[
            {
                "feature_name": "income",
                "psi": 0.05,
                "ks_statistic": 0.375,
                "ks_pvalue": None,
                "js_divergence": 0.025,
                "weighted_score": 0.25,
                "severity": "green",
            }
        ],
        output_drift=None,
        recommendations=[],
        red_count=0,
        yellow_count=0,
        green_count=1,
    )
    assert "| Feature | PSI | KS Statistic |" in markdown
    assert "Overall Health Score: 83.3/100 (GREEN)" in markdown
    assert "| income | 0.050 | 0.375 | 0.025 | 0.250 | green |" in markdown
    assert "p-value" not in markdown
    assert "Correlation" not in markdown


def test_statistical_migration_discards_only_invalid_derived_rows() -> None:
    path = Path("migrations/versions/0003_statistical_correctness.py")
    spec = spec_from_file_location("statistical_migration", path)
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.down_revision == "0002_reliability"
    with patch.object(migration.op, "execute") as execute:
        migration.upgrade()
        assert [call.args[0] for call in execute.call_args_list] == [
            "DELETE FROM drift_scores",
            "DELETE FROM health_reports",
        ]
        execute.reset_mock()
        migration.downgrade()
        execute.assert_not_called()


def test_no_evidence_health_remains_null_in_response_and_markdown() -> None:
    markdown = _render_markdown_report(
        model_name="test-model",
        week_start=date(2026, 3, 16),
        week_end=date(2026, 3, 22),
        overall_score=None,
        ranked_rows=[],
        output_drift=None,
        recommendations=[],
        red_count=0,
        yellow_count=0,
        green_count=0,
    )
    response = HealthReportResponse(
        id=uuid4(),
        model_id=uuid4(),
        week_start=date(2026, 3, 16),
        week_end=date(2026, 3, 22),
        overall_score=None,
        report_json={"executive_summary": {"overall_score": None}},
        report_markdown=markdown,
        generated_at=datetime.now(timezone.utc),
    ).model_dump(mode="json")
    assert response["overall_score"] is None
    assert response["report_json"]["executive_summary"]["overall_score"] is None
    assert "Overall Health Score: No drift data available" in markdown
    assert "100" not in markdown
