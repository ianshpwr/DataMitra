"""
DataMitra — LangGraph Pipeline
Wires DataAgent → InsightAgent → CriticAgent into a state machine
with conditional routing and retry logic.
"""
import polars as pl
from typing import Optional, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from agents.data_agent.agent    import DataProcessingAgent
from agents.data_agent.models   import DataContext
from agents.insight_agent.agent import InsightAgent
from agents.insight_agent.models import InsightBundle
from agents.critic_agent.agent  import CriticAgent
from agents.critic_agent.models import CriticReport
from agents.decision_agent.agent  import DecisionAgent
from agents.decision_agent.models import DecisionPack
# ── Pipeline state definition ─────────────────────────────────────────────────
# Everything the pipeline needs to pass between nodes lives here.
# LangGraph serialises this between steps.

class PipelineState(TypedDict):
    # Input
    source_type:    str                    # "static" | "live"
    file_path:      Optional[str]          # for static files
    use_sample:     bool

    # Agent outputs (filled progressively)
    clean_df:       Optional[pl.DataFrame]
    data_ctx:       Optional[DataContext]
    raw_bundle:     Optional[InsightBundle]
    final_bundle:   Optional[InsightBundle]
    critic_report:  Optional[CriticReport]
    decision_pack:  Optional[Any] 

    # Control flow
    retry_count:    int
    error:          Optional[str]
    needs_review:   bool
    completed:      bool


# ── Node functions ────────────────────────────────────────────────────────────

_data_agent    = DataProcessingAgent()
_insight_agent = InsightAgent()
_critic_agent  = CriticAgent()
_decision_agent = DecisionAgent()



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

        return {**state, "clean_df": clean_df, "data_ctx": ctx, "error": None}

    except Exception as e:
        return {**state, "error": f"DataAgent failed: {e}"}


def run_insight_agent(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: insight_agent")
    try:
        bundle = _insight_agent.run(state["clean_df"], state["data_ctx"])
        return {**state, "raw_bundle": bundle}
    except Exception as e:
        return {**state, "error": f"InsightAgent failed: {e}"}


def run_critic_agent(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: critic_agent")
    try:
        validated_bundle, report = _critic_agent.run(
            state["raw_bundle"],
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


def finalize_output(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: output — pipeline complete")
    return {**state, "completed": True}


def flag_for_review(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: human_review — flagging low-confidence insights")
    # In production: push to a review queue / send a Slack alert
    # For now: mark completed but with review flag
    return {**state, "completed": True}


# ── Routing logic ─────────────────────────────────────────────────────────────

MAX_RETRIES = 3

def route_after_critic(state: PipelineState) -> str:
    """
    Decides where to go after the Critic Agent runs.
    Returns a node name — LangGraph uses this to pick the next step.
    """
    if state.get("error"):
        return "output"   # fail gracefully with whatever we have

    report = state.get("critic_report")
    if report is None:
        return "output"

    retry_count = state.get("retry_count", 0)

    # Retry path: Critic says quality is low AND we haven't hit max retries
    if report.retry_recommended and retry_count < MAX_RETRIES:
        print(f"[Pipeline] Router: retry ({retry_count + 1}/{MAX_RETRIES}) — "
              f"{report.retry_reason}")
        state["retry_count"] = retry_count + 1
        return "insight_agent"   # re-run insight agent with same data

    # Human review path: passed but low confidence on several insights
    if report.needs_human_review:
        return "human_review"

    # Happy path
    return "output"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("data_agent",    run_data_agent)
    graph.add_node("insight_agent", run_insight_agent)
    graph.add_node("critic_agent",  run_critic_agent)
    graph.add_node("decision_agent", run_decision_agent)
    graph.add_node("output",        finalize_output)
    graph.add_node("human_review",  flag_for_review)

    # Edges
    graph.set_entry_point("data_agent")
    graph.add_edge("data_agent",    "insight_agent")
    graph.add_edge("insight_agent", "critic_agent")
    graph.add_edge("output",        END)
    graph.add_edge("human_review",  END)

    # Conditional routing after Critic
    graph.add_conditional_edges(
            "critic_agent",
            route_after_critic,
            {
                "insight_agent":  "insight_agent",
                "output":         "decision_agent",    # changed: critic → decision
                "human_review":   "decision_agent",    # changed: review → decision
            },
        )
    graph.add_edge("decision_agent", "output")


    return graph.compile()


# ── Public entry point ────────────────────────────────────────────────────────

def run_static_pipeline(file_path: str, use_sample: bool = False) -> PipelineState:
    pipeline = build_pipeline()
    initial_state: PipelineState = {
        "source_type":  "static",
        "file_path":    file_path,
        "use_sample":   use_sample,
        "clean_df":     None,
        "data_ctx":     None,
        "raw_bundle":   None,
        "final_bundle": None,
        "critic_report": None,
        "retry_count":  0,
        "error":        None,
        "needs_review": False,
        "completed":    False,
    }
    return pipeline.invoke(initial_state)


def run_live_pipeline() -> PipelineState:
    pipeline = build_pipeline()
    initial_state: PipelineState = {
        "source_type":  "live",
        "file_path":    None,
        "use_sample":   False,
        "clean_df":     None,
        "data_ctx":     None,
        "raw_bundle":   None,
        "final_bundle": None,
        "critic_report": None,
        "retry_count":  0,
        "error":        None,
        "needs_review": False,
        "completed":    False,
        
    }
    return pipeline.invoke(initial_state)

def run_decision_agent(state: PipelineState) -> PipelineState:
    print(f"[Pipeline] Node: decision_agent")
    try:
        pack = _decision_agent.run(
            state["final_bundle"],
            state["data_ctx"],
        )
        return {**state, "decision_pack": pack}
    except Exception as e:
        # Decision agent failure is non-fatal — pipeline continues
        print(f"[Pipeline] Warning: DecisionAgent failed: {e}")
        return {**state, "decision_pack": None}