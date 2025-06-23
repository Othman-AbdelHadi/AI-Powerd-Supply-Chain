# chains/agent_flow.py

from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda
from typing import TypedDict, List, Any

from agents.demand_agent import DemandAgent
from agents.supplier_agent import SupplierAgent
from agents.delay_agent import DelayRootCauseAgent
from agents.recommendation_agent import RecommendationAgent

# === Shared State ===
class AgentState(TypedDict):
    df: Any
    demand_data: List[int]
    forecast: List[int]
    supplier_eval: Any
    delay_causes: dict
    recommendations: List[str]

# === Agent Functions ===
def run_demand_agent(state: AgentState) -> AgentState:
    forecast = DemandAgent(state["demand_data"]).forecast_next_days()
    return {**state, "forecast": forecast}

def run_supplier_agent(state: AgentState) -> AgentState:
    agent = SupplierAgent(state["df"])
    return {**state, "supplier_eval": agent.evaluate()}

def run_delay_agent(state: AgentState) -> AgentState:
    agent = DelayRootCauseAgent(state["df"])
    return {**state, "delay_causes": agent.analyze()}

def run_recommendation_agent(state: AgentState) -> AgentState:
    agent = RecommendationAgent(state["df"])
    return {**state, "recommendations": agent.recommend()}

# === Define Agent Flow ===
def get_agent_flow():
    graph = StateGraph(AgentState)

    graph.add_node("ForecastDemand", RunnableLambda(run_demand_agent))
    graph.add_node("EvaluateSuppliers", RunnableLambda(run_supplier_agent))
    graph.add_node("AnalyzeDelays", RunnableLambda(run_delay_agent))
    graph.add_node("RecommendActions", RunnableLambda(run_recommendation_agent))

    graph.set_entry_point("ForecastDemand")
    graph.add_edge("ForecastDemand", "EvaluateSuppliers")
    graph.add_edge("EvaluateSuppliers", "AnalyzeDelays")
    graph.add_edge("AnalyzeDelays", "RecommendActions")
    graph.add_edge("RecommendActions", END)

    return graph.compile()