"""Deterministic model health investigation over an MCP session."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.schemas.ingest import PREDICTIONS_FEATURE_NAME
from model_health_agent.narration import narrate_report


class ToolError(Exception):
    """A coded error returned by an MCP tool."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class State(TypedDict, total=False):
    model_id: str
    health: dict
    findings: list[dict]
    report: dict


def triage(state: State) -> str:
    return (
        "investigate"
        if state["health"]["status"] in ("critical", "attention")
        else "summarize"
    )


def mean_direction(direction):
    if direction in ("above", "below"):
        return f"the window mean is {direction} the reference"
    if direction == "equal":
        return "the window mean equals the reference"
    return "the window mean direction is unavailable"


def build_graph(session, narrate=None):
    """Build a graph using an initialized MCP session."""

    async def call(name: str, **arguments: Any) -> dict:
        result = await session.call_tool(name, arguments)
        if result.isError:
            error = (result.structuredContent or {}).get("error", {})
            raise ToolError(
                error.get("code", "api_error"),
                error.get("message", str(result.content)),
            )
        return result.structuredContent

    async def collect(state: State) -> dict:
        return {
            "health": await call("analyze_model_health", model_id=state["model_id"])
        }

    async def investigate(state: State) -> dict:
        health = state["health"]
        names = [row["feature_name"] for row in health["problem_features"][:3]]
        if health["prediction_severity"] not in (None, "green"):
            names.append(PREDICTIONS_FEATURE_NAME)
        findings = []
        for name in names:
            args = {
                "model_id": state["model_id"],
                "window_date": health["latest_window"],
            }
            drift = await call(
                "get_feature_drift",
                **args,
                **({} if name == PREDICTIONS_FEATURE_NAME else {"feature_name": name}),
            )
            distribution = await call(
                "get_feature_distribution", **args, feature_name=name
            )
            reference = (distribution["reference_stats"] or {}).get("mean")
            window = (distribution["window_stats"] or {}).get("mean")
            direction = None
            if reference is not None and window is not None:
                direction = (
                    "above"
                    if window > reference
                    else "below" if window < reference else "equal"
                )
            findings.append(
                {
                    "feature_name": name,
                    "drift": (
                        drift["prediction"]
                        if name == PREDICTIONS_FEATURE_NAME
                        else drift["features"][0]
                    ),
                    "reference_mean": reference,
                    "window_mean": window,
                    "direction": direction,
                    "alerts": [
                        a
                        for a in health["active_alerts"]
                        if a.get("feature_name") == name
                    ],
                }
            )
        return {"findings": findings}

    async def summarize(state: State) -> dict:
        return {"findings": []}

    async def report(state: State) -> dict:
        health = state["health"]
        status = health["status"]
        recommendations = []
        if status in ("critical", "attention"):
            for finding in state["findings"]:
                if finding["feature_name"] == PREDICTIONS_FEATURE_NAME:
                    continue
                recommendations.append(
                    f"Check the upstream data source for {finding['feature_name']}; {mean_direction(finding['direction'])}."
                )
            recommendations.append(
                "Evaluate performance on recent labeled data before deciding whether to retrain."
            )
        if health["prediction_severity"] not in (None, "green"):
            prediction = next(
                (
                    f
                    for f in state["findings"]
                    if f["feature_name"] == PREDICTIONS_FEATURE_NAME
                ),
                None,
            )
            direction = prediction["direction"] if prediction else None
            detail = mean_direction(direction)
            recommendations.append(
                f"Review downstream decisions that use the score distribution; {detail}."
            )
        if not health["report_covers_latest_window"]:
            recommendations.append("Generate a report for the latest window.")
        if status == "no_data":
            recommendations.append(
                "Ingest a snapshot; no conclusion can be drawn without data."
            )
        elif status == "unknown":
            recommendations.append(
                "Investigate missing severity evidence; no health conclusion can be drawn."
            )
        elif status == "healthy":
            recommendations.append("Continue monitoring.")
        result = {
            "model": health["model"],
            "status": status,
            "evidence": {
                "window": health["latest_window"],
                "severity_counts": health["severity_counts"],
                "prediction_severity": health["prediction_severity"],
                "overall_score": health["overall_score"],
                "report_covers_latest_window": health["report_covers_latest_window"],
            },
            "findings": state["findings"],
            "stable_features": health["stable_features"],
            "active_alerts": health["active_alerts"],
            "recommendations": recommendations,
            "limitations": health["limitations"],
            "mode": "offline",
            "narrative": None,
        }
        return {"report": await (narrate or narrate_report)(result)}

    # Synchronous routing can hang when the graph runs repeatedly.
    async def route(state: State) -> str:
        return triage(state)

    graph = StateGraph(State)
    graph.add_node("collect", collect)
    graph.add_node("investigate", investigate)
    graph.add_node("summarize", summarize)
    graph.add_node("report", report)
    graph.add_edge(START, "collect")
    graph.add_conditional_edges(
        "collect", route, {"investigate": "investigate", "summarize": "summarize"}
    )
    graph.add_edge("investigate", "report")
    graph.add_edge("summarize", "report")
    graph.add_edge("report", END)
    return graph.compile()
