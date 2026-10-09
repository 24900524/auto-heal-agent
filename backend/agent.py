
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")


class AgentState(TypedDict):
    health: dict
    logs: list
    diagnosis: str
    decision: str
    action: str
    verified: bool
    final_status: str


llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=API_KEY,
    temperature=0,
)


def observe(state: AgentState):
    return {
        "diagnosis": "",
        "decision": "",
        "action": "none",
        "verified": False,
        "final_status": "pending",
    }


def diagnose(state: AgentState):
    prompt = f"""
You are an incident diagnosis agent.

Analyze the system evidence:
Health: {state['health']}
Logs: {state['logs']}

Choose exactly one decision:
REPAIR — only if evidence clearly indicates the simulated
database connection-pool fault.
ESCALATE — if evidence is insufficient, the fault is unknown,
or the problem is unsafe to repair automatically.

Return these lines:
CAUSE: <likely cause>
EVIDENCE: <supporting evidence>
DECISION: <REPAIR or ESCALATE>
"""
    response = llm.invoke(prompt)
    result = str(response.content)

    
    decision = "ESCALATE"
    upper_result = result.upper()

    if (
        "DECISION: REPAIR" in upper_result
        and (
            "CONNECTION-POOL" in upper_result
            or "CONNECTION POOL" in upper_result
        )
    ):
        decision = "REPAIR"

    return {"diagnosis": result, "decision": decision}


def decide(state: AgentState):
    if state["decision"] == "REPAIR":
        return "repair"
    return "escalate"


def repair(state: AgentState):
    # This node only authorizes a simulated repair.
    # It does not execute arbitrary commands.
    return {"action": "simulated_connection_pool_reset"}


def verify(state: AgentState):
    # The API layer performs the repair and supplies
    # the actual health result in a later integration step.
    return {
        "verified": False,
        "final_status": "repair_pending_backend_integration",
    }


def escalate(state: AgentState):
    return {
        "action": "human_review_required",
        "verified": False,
        "final_status": "escalated",
    }


graph = StateGraph(AgentState)
graph.add_node("observe", observe)
graph.add_node("diagnose", diagnose)
graph.add_node("repair", repair)
graph.add_node("verify", verify)
graph.add_node("escalate", escalate)

graph.add_edge(START, "observe")
graph.add_edge("observe", "diagnose")
graph.add_conditional_edges(
    "diagnose",
    decide,
    {"repair": "repair", "escalate": "escalate"},
)
graph.add_edge("repair", "verify")
graph.add_edge("verify", END)
graph.add_edge("escalate", END)

agent_graph = graph.compile()


def diagnose_incident(health: dict, logs: list) -> str:
    """Compatibility function for the existing FastAPI backend."""
    result = agent_graph.invoke({
        "health": health,
        "logs": logs,
        "diagnosis": "",
        "decision": "",
        "action": "none",
        "verified": False,
        "final_status": "pending",
    })
    return (
        f"{result['diagnosis']}\n"
        f"Agent decision: {result['decision']}\n"
        f"Action: {result['action']}\n"
        f"Status: {result['final_status']}"
    )

