"""FastMCP server exposing MedAssist's structured medical tools.

Run standalone (stdio transport, for MCP clients like Claude Desktop):
    uv run python mcp_server.py

Or serve over HTTP:
    uv run fastmcp run mcp_server.py:mcp --transport http --port 8001

The LangGraph backend (BackEnd.py) also imports `mcp` and calls the
tools in-process, so no separate server process is needed for the app.
"""

from ddgs import DDGS
from fastmcp import FastMCP

mcp = FastMCP("MedAssist-Tools")


# Symptom keywords that should always escalate to emergency care.
RED_FLAG_SYMPTOMS = {
    "chest pain": "Possible cardiac event — seek emergency care immediately.",
    "shortness of breath": "Breathing difficulty can be life-threatening — seek urgent care.",
    "difficulty breathing": "Breathing difficulty can be life-threatening — seek urgent care.",
    "severe bleeding": "Uncontrolled bleeding requires emergency care.",
    "slurred speech": "Possible stroke sign (FAST) — call emergency services now.",
    "face drooping": "Possible stroke sign (FAST) — call emergency services now.",
    "loss of consciousness": "Requires immediate emergency evaluation.",
    "stiff neck with fever": "Possible meningitis — seek emergency care immediately.",
    "worst headache of my life": "Sudden severe headache needs emergency evaluation.",
    "suicidal": "Contact a crisis helpline or emergency services immediately.",
}

# Minimal local formulary of common OTC medications (standard label info).
OTC_MEDICATIONS = {
    "paracetamol": {
        "also_known_as": ["acetaminophen", "tylenol", "crocin", "dolo"],
        "class": "analgesic / antipyretic",
        "typical_adult_dose": "500-1000 mg every 4-6 hours as needed",
        "max_daily_dose": "4000 mg (3000 mg if liver disease or regular alcohol use)",
        "cautions": ["liver disease", "chronic alcohol use"],
    },
    "ibuprofen": {
        "also_known_as": ["advil", "brufen", "nurofen"],
        "class": "NSAID (anti-inflammatory / analgesic)",
        "typical_adult_dose": "200-400 mg every 4-6 hours with food",
        "max_daily_dose": "1200 mg OTC without medical supervision",
        "cautions": ["stomach ulcers", "kidney disease", "aspirin/NSAID allergy", "pregnancy (3rd trimester)"],
    },
    "cetirizine": {
        "also_known_as": ["zyrtec", "alerid"],
        "class": "second-generation antihistamine",
        "typical_adult_dose": "10 mg once daily",
        "max_daily_dose": "10 mg",
        "cautions": ["may cause drowsiness", "kidney impairment"],
    },
    "loperamide": {
        "also_known_as": ["imodium"],
        "class": "antidiarrheal",
        "typical_adult_dose": "4 mg initially, then 2 mg after each loose stool",
        "max_daily_dose": "8 mg OTC",
        "cautions": ["bloody diarrhea", "fever", "do not exceed labeled dose"],
    },
}

# Allergy groups: an allergy to the key conflicts with every drug listed.
ALLERGY_CROSS_REACTIONS = {
    "aspirin": ["ibuprofen"],
    "nsaid": ["ibuprofen"],
    "nsaids": ["ibuprofen"],
}


@mcp.tool
def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """Calculate Body Mass Index and return the WHO category."""
    bmi = weight_kg / (height_cm / 100) ** 2
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal weight"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"
    return {"bmi": round(bmi, 1), "category": category}


@mcp.tool
def check_symptom_red_flags(symptoms: list[str]) -> dict:
    """Check reported symptoms against known emergency red flags."""
    flagged: dict[str, str] = {}
    for symptom in symptoms:
        lowered = symptom.lower()
        for flag, advice in RED_FLAG_SYMPTOMS.items():
            if flag in lowered:
                flagged[symptom] = advice
    return {"red_flags_found": bool(flagged), "details": flagged}


@mcp.tool
def medication_info(name: str) -> dict:
    """Look up label information for a common OTC medication by name or brand."""
    query = name.strip().lower()
    for generic, info in OTC_MEDICATIONS.items():
        if query == generic or query in info["also_known_as"]:
            return {"found": True, "generic_name": generic, **info}
    return {
        "found": False,
        "note": f"'{name}' is not in the local formulary; advise consulting a pharmacist.",
    }


@mcp.tool
def check_allergy_conflict(medication: str, allergies: list[str]) -> dict:
    """Check whether a medication conflicts with the patient's known allergies."""
    med = medication.strip().lower()
    conflicts: list[str] = []
    for allergy in allergies:
        lowered = allergy.strip().lower()
        if not lowered or lowered == "none":
            continue
        if lowered in med or med in lowered:
            conflicts.append(allergy)
        elif med in ALLERGY_CROSS_REACTIONS.get(lowered, []):
            conflicts.append(f"{allergy} (cross-reaction risk)")
    return {"conflict": bool(conflicts), "conflicting_allergies": conflicts}


@mcp.tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web via DuckDuckGo for up-to-date information."""
    try:
        results = DDGS().text(query, max_results=max_results)
        return {
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        }
    except Exception as exc:
        return {"results": [], "error": f"Search failed: {exc}"}


if __name__ == "__main__":
    mcp.run()
