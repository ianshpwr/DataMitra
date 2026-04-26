"""
DataMitra — LangGraph Pipeline
Wires DataAgent → InsightAgent → CriticAgent → DecisionAgent → ChartAgent
"""
import polars as pl
import tempfile
import os
from typing import Optional, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from agents.chart_agent.agent     import ChartSelectorAgent
from agents.data_agent.agent      import DataProcessingAgent
from agents.data_agent.models     import DataContext
from agents.insight_agent.agent   import InsightAgent
from agents.insight_agent.models  import InsightBundle
from agents.critic_agent.agent    import CriticAgent
from agents.critic_agent.models   import CriticReport
from agents.decision_agent.agent  import DecisionAgent
from agents.decision_agent.models import DecisionPack


# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # Input
    source_type:   str
    file_path:     Optional[str]
    use_sample:    bool

    # Agent outputs
    clean_df:      Optional[pl.DataFrame]
    data_ctx:      Optional[DataContext]
    raw_bundle:    Optional[InsightBundle]
    final_bundle:  Optional[InsightBundle]
    critic_report: Optional[CriticReport]
    decision_pack: Optional[Any]
    chart_plans:   Optional[list]
    df_path:       Optional[str]

    # Control flow
    retry_count:   int
    error:         Optional[str]
    needs_review:  bool
    completed:     bool


# ── Agent instances ───────────────────────────────────────────────────────────

_data_agent     = DataProcessingAgent()
_insight_agent  = InsightAgent()
_critic_agent   = CriticAgent()
_decision_agent = DecisionAgent()
_chart_agent    = ChartSelectorAgent()


# ── Node functions ────────────────────────────────────────────────────────────
def run_data_agent(state: PipelineState) -> PipelineState:
    print(f"\n[Pipeline] Node: data_agent (retry={state['retry_count']})")
    try:
        if state["source_type"] == "static":
            clean_df, ctx = _data_agent.run_static(
                state["file_path"],
                use_sample=state.get("use_sample", False),
            )
        else:
            result = _data_agent.run_live_batch()
            if result is None:
                return {**state, "error": "No live data available"}
            clean_df, ctx = result

        # Save parquet immediately so chart_agent can use it later
        df_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False, prefix="datamitra_"
            )
            clean_df.write_parquet(tmp.name)
            df_path = tmp.name
            print(f"  [DataAgent] Saved parquet: {df_path}")
        except Exception as e:
            print(f"  [DataAgent] Warning: could not save parquet: {e}")

        return {
            **state,
            "clean_df": clean_df,
            "data_ctx": ctx,
            "df_path":  df_path,
            "error":    None,
        }

    except Exception as e:
        return {**state, "error": f"DataAgent failed: {e}"}
    


def run_insight_agent(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: insight_agent")
    try:
        bundle = _insight_agent.run(state["clean_df"], state["data_ctx"])
        if bundle is None:
            return {**state, "error": "InsightAgent returned None bundle"}
        return {**state, "raw_bundle": bundle}
    except Exception as e:
        return {**state, "error": f"InsightAgent failed: {e}"}


def run_critic_agent(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: critic_agent")

    raw_bundle = state.get("raw_bundle")
    if raw_bundle is None:
        # InsightAgent failed upstream — skip critic, propagate the error
        print("  [CriticAgent] Skipped — raw_bundle is None (upstream error)")
        return {
            **state,
            "final_bundle":  None,
            "critic_report": None,
            "needs_review":  False,
        }

    try:
        validated_bundle, report = _critic_agent.run(
            raw_bundle,
            state["clean_df"],
            state["data_ctx"],
        )
        return {
            **state,
            "final_bundle":  validated_bundle,
            "critic_report": report,
            "needs_review":  report.needs_human_review,
        }
    except Exception as e:
        return {**state, "error": f"CriticAgent failed: {e}"}


def run_decision_agent(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: decision_agent")
    try:
        pack = _decision_agent.run(
            state["final_bundle"],
            state["data_ctx"],
        )
        return {**state, "decision_pack": pack}
    except Exception as e:
        print(f"[Pipeline] Warning: DecisionAgent failed: {e}")
        return {**state, "decision_pack": None}


def run_chart_agent(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: chart_agent")
    df_path = state.get("df_path")
    bundle  = state.get("final_bundle")
    if not df_path or not bundle:
        return {**state, "chart_plans": []}
    try:
        plans = _chart_agent.run(bundle, df_path)
        return {**state, "chart_plans": plans}
    except Exception as e:
        print(f"[Pipeline] Warning: ChartAgent failed: {e}")
        return {**state, "chart_plans": []}


def finalize_output(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: output — pipeline complete")
    return {**state, "completed": True}


def flag_for_review(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: human_review — flagging low-confidence insights")
    return {**state, "completed": True}


# ── Routing ───────────────────────────────────────────────────────────────────

MAX_RETRIES = 3


def route_after_critic(state: PipelineState) -> str:
    if state.get("error"):
        return "decision_agent"

    report      = state.get("critic_report")
    retry_count = state.get("retry_count", 0)

    if report and report.retry_recommended and retry_count < MAX_RETRIES:
        print(f"[Pipeline] Router: retry ({retry_count + 1}/{MAX_RETRIES})")
        state["retry_count"] = retry_count + 1
        return "insight_agent"

    return "decision_agent"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)   # ← was missing

    graph.add_node("data_agent",     run_data_agent)
    graph.add_node("insight_agent",  run_insight_agent)
    graph.add_node("critic_agent",   run_critic_agent)
    graph.add_node("decision_agent", run_decision_agent)
    graph.add_node("chart_agent",    run_chart_agent)
    graph.add_node("output",         finalize_output)
    graph.add_node("human_review",   flag_for_review)

    graph.set_entry_point("data_agent")
    graph.add_edge("data_agent",    "insight_agent")
    graph.add_edge("insight_agent", "critic_agent")
    graph.add_edge("decision_agent","chart_agent")
    graph.add_edge("output",        END)
    graph.add_edge("human_review",  END)

    graph.add_conditional_edges(
        "critic_agent",
        route_after_critic,
        {
            "insight_agent":  "insight_agent",
            "decision_agent": "decision_agent",
        },
    )

    graph.add_conditional_edges(
        "chart_agent",
        lambda s: "human_review" if s.get("needs_review") else "output",
        {
            "output":       "output",
            "human_review": "human_review",
        },
    )

    return graph.compile()


# ── Public entry points ───────────────────────────────────────────────────────

def _initial(source_type: str, file_path: str = None,
             use_sample: bool = False) -> PipelineState:
    return {
        "source_type":   source_type,
        "file_path":     file_path,
        "use_sample":    use_sample,
        "clean_df":      None,
        "data_ctx":      None,
        "raw_bundle":    None,
        "final_bundle":  None,
        "critic_report": None,
        "decision_pack": None,
        "chart_plans":   [],
        "df_path":       None,
        "retry_count":   0,
        "error":         None,
        "needs_review":  False,
        "completed":     False,
    }


def run_static_pipeline(file_path: str, use_sample: bool = False) -> PipelineState:
    return build_pipeline().invoke(_initial("static", file_path, use_sample))


def run_live_pipeline() -> PipelineState:
    return build_pipeline().invoke(_initial("live"))