import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.generator import detect_context, NarrativeGenerator, format_narrative_text, parse_narrative_parts


def test_detects_academic_context():
    text = "I have an exam tomorrow and I keep thinking I'm going to fail."
    ctx = detect_context(text)
    assert ctx["context_name"] == "Exam Preparation & Academic Study"
    assert "student" in ctx["character"]["description"]
    assert "textbooks" in ctx["scenes"][0]["visual"] or "desk" in ctx["scenes"][0]["visual"]


def test_detects_workplace_context():
    text = "My presentation to the boss tomorrow will be a total disaster."
    ctx = detect_context(text)
    assert ctx["context_name"] == "Workplace & Career Performance"
    assert "professional" in ctx["character"]["description"]


def test_detects_relational_context():
    text = "I feel so lonely and like no one cares if I disappear."
    ctx = detect_context(text)
    assert ctx["context_name"] == "Interpersonal Relationships & Social Connection"


def test_fallback_generates_three_grounded_scenes():
    # Test _build_structured_story directly without loading heavy model
    gen = object.__new__(NarrativeGenerator)
    cf = CaseFrame(
        raw_text="I have an exam tomorrow and I keep thinking I'm going to fail.",
        distortions=["catastrophizing", "fortune_telling"],
        core_emotion="fear"
    )
    technique = {
        "id": "decatastrophizing",
        "name": "Decatastrophizing",
        "description": "Examining realistic outcomes versus worst-case fears."
    }

    story = gen._build_structured_story("", cf, technique)

    assert "character" in story
    assert len(story["character"]["appearance"]) > 0
    assert len(story["character"]["clothing"]) > 0

    scenes = story["scenes"]
    assert len(scenes) == 3
    assert scenes[0]["stage"] == "problem"
    assert scenes[1]["stage"] == "reframing"
    assert scenes[2]["stage"] == "resolution"

    # Verify bubble types
    assert scenes[0]["bubble_type"] == "thought"
    assert scenes[1]["bubble_type"] == "thought"
    assert scenes[2]["bubble_type"] == "speech"

    # Verify dialogues are present and non-empty
    for s in scenes:
        assert len(s["dialogue"]) > 0
        assert len(s["visual_description"]) > 0
        assert len(s["narrative"]) > 0

    # Verify voice script is meaningful and non-empty
    voice_script = story["voice_script"]
    assert len(voice_script) > 50


def test_format_narrative_text():
    story = {
        "scenes": [
            {"title": "1. PROBLEM", "narrative": "Feeling exam stress.", "dialogue": "I will fail.", "bubble_type": "thought"},
            {"title": "2. REFRAMING", "narrative": "A test does not define me.", "dialogue": "I can prepare.", "bubble_type": "thought"},
            {"title": "3. RESOLUTION", "narrative": "Study one hour now.", "dialogue": "Doing chapter 1.", "bubble_type": "speech"}
        ]
    }
    formatted = format_narrative_text(story)
    assert "1. PROBLEM:" in formatted
    assert "2. REFRAMING:" in formatted
    assert "3. RESOLUTION:" in formatted
    assert "[Thought Bubble: \"I will fail.\"]" in formatted


def test_parse_narrative_parts_backwards_compatible():
    text = "SCENE 1 - PROBLEM: Stressed.\nSCENE 2 - REFRAME: Calm.\nSCENE 3 - RESOLUTION: Action."
    parts = parse_narrative_parts(text)
    assert "problem" in parts
    assert "reframing" in parts
    assert "resolution" in parts
