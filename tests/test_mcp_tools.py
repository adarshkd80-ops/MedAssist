"""Unit tests for the pure MCP tools in mcp_server.py.

web_search is excluded: it hits the live DuckDuckGo API and would make
CI flaky.
"""

import pytest

from mcp_server import (
    calculate_bmi,
    check_allergy_conflict,
    check_symptom_red_flags,
    medication_info,
)


class TestCalculateBmi:
    def test_normal_weight(self):
        result = calculate_bmi(weight_kg=70, height_cm=175)
        assert result == {"bmi": 22.9, "category": "normal weight"}

    def test_underweight(self):
        assert calculate_bmi(weight_kg=45, height_cm=175)["category"] == "underweight"

    def test_overweight(self):
        assert calculate_bmi(weight_kg=80, height_cm=170)["category"] == "overweight"

    def test_obese(self):
        assert calculate_bmi(weight_kg=100, height_cm=170)["category"] == "obese"

    def test_category_boundary_25_is_overweight(self):
        # BMI exactly 25.0: 76.5625 kg at 175 cm
        result = calculate_bmi(weight_kg=76.5625, height_cm=175)
        assert result["category"] == "overweight"


class TestCheckSymptomRedFlags:
    def test_flags_chest_pain(self):
        result = check_symptom_red_flags(symptoms=["crushing chest pain"])
        assert result["red_flags_found"] is True
        assert "crushing chest pain" in result["details"]

    def test_case_insensitive(self):
        result = check_symptom_red_flags(symptoms=["Severe BLEEDING from arm"])
        assert result["red_flags_found"] is True

    def test_benign_symptoms_pass(self):
        result = check_symptom_red_flags(symptoms=["mild headache", "runny nose"])
        assert result == {"red_flags_found": False, "details": {}}

    def test_empty_list(self):
        assert check_symptom_red_flags(symptoms=[])["red_flags_found"] is False


class TestMedicationInfo:
    def test_lookup_by_generic_name(self):
        result = medication_info(name="paracetamol")
        assert result["found"] is True
        assert result["generic_name"] == "paracetamol"

    def test_lookup_by_brand_name(self):
        result = medication_info(name="Advil")
        assert result["found"] is True
        assert result["generic_name"] == "ibuprofen"

    def test_unknown_medication(self):
        result = medication_info(name="unobtainium")
        assert result["found"] is False
        assert "pharmacist" in result["note"]


class TestCheckAllergyConflict:
    def test_direct_conflict(self):
        result = check_allergy_conflict(
            medication="ibuprofen", allergies=["ibuprofen"]
        )
        assert result["conflict"] is True

    def test_cross_reaction(self):
        result = check_allergy_conflict(medication="ibuprofen", allergies=["aspirin"])
        assert result["conflict"] is True
        assert any("cross-reaction" in c for c in result["conflicting_allergies"])

    def test_no_conflict(self):
        result = check_allergy_conflict(
            medication="paracetamol", allergies=["penicillin"]
        )
        assert result == {"conflict": False, "conflicting_allergies": []}

    def test_none_and_blank_allergies_ignored(self):
        result = check_allergy_conflict(
            medication="ibuprofen", allergies=["none", "  ", ""]
        )
        assert result["conflict"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
