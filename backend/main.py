from backend.agent import agent_graph
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="Auto-Heal Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
service_state = {
    "database_healthy": True,
    "fault_active": False
}

system_logs = []


def add_log(level, message):
    system_logs.append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "message": message
    })


add_log("INFO", "Auto-Heal backend started")


@app.get("/")
def home():
    return {"message": "Auto-Heal backend is running"}


@app.get("/health")
def health():
    healthy = (
        service_state["database_healthy"]
        and not service_state["fault_active"]
    )
    return {
        "status": "healthy" if healthy else "unhealthy",
        **service_state
    }


@app.get("/logs")
def logs():
    return {"total_logs": len(system_logs), "logs": system_logs[-50:]}


@app.post("/inject-fault")
def inject_fault():
    service_state["database_healthy"] = False
    service_state["fault_active"] = True
    add_log("ERROR", "Database connection pool exhausted")
    return {"message": "Fault injected", "status": "unhealthy"}


@app.post("/repair")
def repair():
    if not service_state["fault_active"]:
        return {"message": "No active fault", "verified": True}

    service_state["database_healthy"] = True
    service_state["fault_active"] = False
    add_log("INFO", "Simulated connection pool reset")

    verified = (
        service_state["database_healthy"]
        and not service_state["fault_active"]
    )
    return {
        "message": "Repair completed",
        "verified": verified,
        "status": "healthy" if verified else "unhealthy"
    }


@app.post("/agent/run")
def run_agent():
    # 1. OBSERVE
    health_snapshot = {
        "status": (
            "healthy"
            if service_state["database_healthy"]
            and not service_state["fault_active"]
            else "unhealthy"
        ),
        **service_state,
    }
    recent_logs = system_logs[-10:]

    # 2. DIAGNOSE using the LangGraph workflow
    try:
        result = agent_graph.invoke({
            "health": health_snapshot,
            "logs": recent_logs,
            "diagnosis": "",
            "decision": "",
        })
    except Exception as exc:
        add_log("ERROR", f"Agent diagnosis failed: {type(exc).__name__}")
        return {
            "status": "error",
            "message": "Diagnosis failed. No repair was attempted.",
            "verified": False,
        }

    decision = result["decision"]
    action = "none"

    # 3. ACT only when the agent approves repair
    if decision == "REPAIR":
        if service_state["fault_active"] or not service_state["database_healthy"]:
            service_state["database_healthy"] = True
            service_state["fault_active"] = False
            action = "simulated_connection_pool_reset"
            add_log("INFO", "Agent executed approved simulated repair")
        else:
            add_log("INFO", "No active fault; repair not needed")
    else:
        add_log("WARNING", "Agent requested human review; no repair performed")

    # 4. VERIFY the actual simulated service state
    verified = (
        service_state["database_healthy"]
        and not service_state["fault_active"]
    )

    if action != "none" and verified:
        add_log("INFO", "Agent verified service recovery")
    elif action != "none" and not verified:
        add_log("ERROR", "Agent could not verify service recovery")

    # 5. REPORT
    return {
        "diagnosis": result["diagnosis"],
        "decision": decision,
        "action": action,
        "verified": verified,
        "status": "healthy" if verified else "unhealthy",
        "logs_reviewed": len(recent_logs),
    }
