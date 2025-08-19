# tests/test_router.py
from src.orchestrator.router import KeywordRouter, RouteDecision

def _make_cfg():
    return {
        "router": {
            "confidence_threshold": 0.6,
            "keywords": {
                "incident": ["incidente", "outage", "downtime", "rca", "postmortem", "sev"],
                "career":   ["feedback", "desempeño", "okr", "okrs", "objetivo", "promoción", "30/60/90", "360"],
                "general":  ["general", "definición", "explicación"],
            },
            "greetings": ["hola", "hello", "hi"],
        }
    }

def _make_agents_cfg():
    return [
        {"name": "rag_general",      "domains": ["general", "career", "incident"]},
        {"name": "career_coach",     "domains": ["career", "general"]},
        {"name": "career_planner",   "domains": ["career", "general"]},
        {"name": "incident_analyst", "domains": ["incident", "general"]},
        {"name": "evaluador",        "domains": ["general", "career"]},
    ]

def test_router_greeting_goes_to_general():
    PR = KeywordRouter(_make_cfg(), _make_agents_cfg())
    rd: RouteDecision = PR.route("hola!")
    assert rd.agent in {"rag_general"}, rd
    assert rd.confidence >= 0.6

def test_router_306090_goes_to_planner():
    PR = KeywordRouter(_make_cfg(), _make_agents_cfg())
    rd = PR.route("Quiero un plan 30/60/90 para mi nuevo rol backend.")
    assert rd.agent == "career_planner"
    assert rd.confidence >= 0.9

def test_router_feedback_360_goes_to_coach():
    PR = KeywordRouter(_make_cfg(), _make_agents_cfg())
    rd = PR.route("Necesito feedback 360 para mi evaluación.")
    assert rd.agent == "career_coach"
    assert rd.confidence >= 0.8

def test_router_incident_terms_go_to_incident_analyst():
    PR = KeywordRouter(_make_cfg(), _make_agents_cfg())
    rd = PR.route("Tuvimos un incidente sev2 con downtime, necesito RCA y postmortem.")
    assert rd.agent == "incident_analyst"
    assert rd.confidence >= 0.8

def test_router_ambiguous_falls_back_to_supervisor():
    PR = KeywordRouter(_make_cfg(), _make_agents_cfg())
    rd = PR.route("Quiero mejorar procesos y también objetivos para mi equipo.")
    # puede no superar el umbral; en ese caso agent=None → supervisor
    assert rd.agent in (None, "career_planner", "career_coach")
