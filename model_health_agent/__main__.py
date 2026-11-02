"""Command-line entry point for the model health agent."""

import argparse
import asyncio
import json
import os
import sys
from uuid import UUID

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from model_health_agent.graph import ToolError, build_graph


async def run(model_id):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "driftwatch_mcp"],
        env={
            **os.environ,
            "DRIFTWATCH_URL": os.getenv("DRIFTWATCH_URL", "http://localhost:8000"),
        },
    )
    error = None
    report = None
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            try:
                report = (await build_graph(session).ainvoke({"model_id": model_id}))[
                    "report"
                ]
            except ToolError as exc:
                error = exc
    if error is not None:
        raise error
    return report


def format_number(value, spec):
    return format(value, spec) if value is not None else "not available"


def render_report(report):
    """Render a structured report as readable Markdown."""
    evidence = report["evidence"]
    score = evidence["overall_score"]
    lines = [
        f"# Model health: {report['model'].get('name', report['model'].get('id'))}",
        f"\nStatus: **{report['status']}**",
        "\n## Evidence\n",
        f"- Window: {evidence['window']}",
        "- Severity counts: "
        + ", ".join(
            f"{name}: {count}" for name, count in evidence["severity_counts"].items()
        ),
        f"- Prediction severity: {evidence['prediction_severity']}",
        f"- Overall score: {format_number(score, '.1f')}",
        f"- Report covers window: {'yes' if evidence['report_covers_latest_window'] else 'no'}",
    ]
    lines.append("\n## Findings\n")
    for finding in report["findings"]:
        drift = finding["drift"]
        name = (
            "Model predictions"
            if finding["feature_name"] == "__predictions__"
            else finding["feature_name"]
        )
        count = len(finding["alerts"])
        lines.append(
            f"- {name}: PSI {format_number(drift['psi'], '.3f')} ({drift['severity']}), "
            f"weighted {format_number(drift['weighted_score'], '.3f')}, mean {format_number(finding['reference_mean'], ',.3f')} reference vs "
            f"{format_number(finding['window_mean'], ',.3f')} window ({finding['direction'] or 'unknown'}), "
            f"{count} active alert{'s' if count != 1 else ''}"
        )
    if not report["findings"]:
        lines.append("None.")
    for title, key in (
        ("Stable features", "stable_features"),
        ("Active alerts", "active_alerts"),
        ("Recommendations", "recommendations"),
    ):
        lines.append(f"\n## {title}\n")
        items = report[key]
        lines.extend(
            f"- {item['message'] if key == 'active_alerts' else item}" for item in items
        )
        if not items:
            lines.append("None.")
    lines.append(f"\n## Limitations\n\n{report['limitations'] or 'None.'}")
    if report["narrative"]:
        lines.append(f"\n## Narrative\n\n{report['narrative']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Investigate a DriftWatch model")
    parser.add_argument("model_id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        model_id = str(UUID(args.model_id))
    except ValueError:
        print("Invalid model ID: expected a UUID", file=sys.stderr)
        return 2
    try:
        report = asyncio.run(run(model_id))
    except ToolError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
