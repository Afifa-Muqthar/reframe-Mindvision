import os
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

WORKPLACE_INJUSTICE_INPUT = (
    "My supervisor gave credit for my three-month project to another colleague in the team meeting. "
    "I'm furious and feeling completely powerless."
)

GENERIC_STRESS_PHRASES = [
    "Navigating a challenging personal situation",
    "Personal stress, ambiguity, and self-doubt",
    "Your choices today, treating yourself with patience",
    "Having doubts or feeling unsettled is a normal human experience",
    "Take a deep breath, write down the one single thing",
]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_workplace_injustice_psychological_understanding_endpoint(mock_thread, mock_tts, client):
    """
    Verify that the /process endpoint returns grounded, case-specific psychological understanding
    and ZERO generic fallback boilerplate for the workplace injustice debugging case.
    """
    response = client.post("/process", data={"text": WORKPLACE_INJUSTICE_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    assert "reasoning" in data

    reasoning = data["reasoning"]
    what_happening = reasoning.get("what_may_be_happening", "")
    what_control = reasoning.get("what_you_can_control", "")
    reframe = reasoning.get("reframe", "")
    next_step = reasoning.get("next_step", "")

    # 1. Assert zero generic fallback leakage in reasoning
    all_reasoning_text = f"{what_happening} {what_control} {reframe} {next_step}"
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in all_reasoning_text, f"Generic phrase leaked into UI reasoning: '{generic}'"

    # 2. Assert grounded values are present
    assert "supervisor" in what_happening.lower() or "violation" in what_happening.lower()
    assert "documenting" in what_control.lower() or "mentor" in what_control.lower() or "boundaries" in what_control.lower()
    assert "anger" in reframe.lower() or "unfairness" in reframe.lower()
    assert "document" in next_step.lower() or "record" in next_step.lower()

    # 3. Assert technique is grounded
    assert "Validation" in data["technique"]["name"]
    assert "Greenberg" in data["technique"]["citation"] or "Emotion-Focused" in data["technique"]["citation"]

    # 4. Assert external threat badge flag is true (trauma-informed/external violation)
    assert data["is_external_threat"] is True


LEGACY_REASONING_FIELDS = [
    "situation",
    "primary_emotion",
    "central_conflict",
    "automatic_thought",
    "pattern",
    "coping_strategy",
    "specific_reframe",
    "concrete_next_step",
]

EXPECTED_GROUNDED_FIELDS = {
    "what_may_be_happening",
    "what_you_can_control",
    "what_is_not_controllable",
    "reframe",
    "next_step",
}


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_reasoning_object_contains_only_grounded_fields_and_no_legacy_fields(mock_thread, mock_tts, client):
    """
    Regression test asserting that legacy reasoning fields are strictly absent
    from the final /process reasoning dictionary, and only the 5 intended
    grounded fields are preserved.
    """
    response = client.post("/process", data={"text": WORKPLACE_INJUSTICE_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert "reasoning" in data
    reasoning = data["reasoning"]

    # 1. Assert exact key set matches expected grounded fields
    assert set(reasoning.keys()) == EXPECTED_GROUNDED_FIELDS, (
        f"Reasoning keys mismatch. Got: {set(reasoning.keys())}, Expected: {EXPECTED_GROUNDED_FIELDS}"
    )

    # 2. Explicitly assert every legacy field is absent
    for legacy_key in LEGACY_REASONING_FIELDS:
        assert legacy_key not in reasoning, f"Legacy reasoning field '{legacy_key}' must not be present in reasoning object"

    # 3. Assert all 5 grounded fields have non-empty values
    for grounded_key in EXPECTED_GROUNDED_FIELDS:
        assert reasoning[grounded_key], f"Grounded field '{grounded_key}' must not be empty"
