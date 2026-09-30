"""
Evidence-Informed Strategy Selector and Reframe Synthesizer.

Connects the parsed PsychologicalProfile with the structured Psychological Strategy
Knowledge Base (anchored in APA, SAMHSA, and NIMH frameworks).
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .situation_parser import PsychologicalProfile

_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "knowledge_base",
    "psychological_strategies.json",
)

_strategies_cache = None


def _load_strategies():
    global _strategies_cache
    if _strategies_cache is None:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            _strategies_cache = json.load(f)
    return _strategies_cache


@dataclass
class PsychologicalStrategy:
    id: str
    category: str
    name: str
    citation: str
    mechanism: str
    what_is_controllable: str
    what_is_not_controllable: str
    reframe: str
    concrete_action: str
    voice_script: str
    safety_rule: str
    is_external_threat: bool
    comic_archetype: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "name": self.name,
            "citation": self.citation,
            "mechanism": self.mechanism,
            "what_is_controllable": self.what_is_controllable,
            "what_is_not_controllable": self.what_is_not_controllable,
            "reframe": self.reframe,
            "concrete_action": self.concrete_action,
            "voice_script": self.voice_script,
            "safety_rule": self.safety_rule,
            "is_external_threat": self.is_external_threat,
            "comic_archetype": self.comic_archetype,
        }


def select_strategy(profile: PsychologicalProfile) -> PsychologicalStrategy:
    """
    Selects the optimal evidence-based strategy matching the situation and
    generates tailored reframes and concrete actions.
    """
    strategies = _load_strategies()

    # Match by applicable situation
    chosen = None
    for s in strategies:
        if profile.situation_id in s.get("applicable_situations", []):
            chosen = s
            break

    # Fallback to category or general
    if not chosen:
        for s in strategies:
            if s.get("category") == profile.category:
                chosen = s
                break

    if not chosen:
        chosen = strategies[-1]  # general_grounded_perspective

    # Determine reframe and action
    reframe_text = chosen.get("reframe_template", "")
    action_text = chosen.get("action_template", "")
    voice_script = chosen.get("voice_story_template", "")

    # Controllability synthesis
    controllable = profile.controllable or chosen.get("what_is_controllable", "")
    uncontrollable = profile.uncontrollable or chosen.get("what_is_not_controllable", "")

    return PsychologicalStrategy(
        id=chosen["id"],
        category=chosen.get("category", "cognitive"),
        name=chosen["name"],
        citation=chosen["citation"],
        mechanism=chosen.get("mechanism", ""),
        what_is_controllable=controllable,
        what_is_not_controllable=uncontrollable,
        reframe=reframe_text,
        concrete_action=action_text,
        voice_script=voice_script,
        safety_rule=chosen.get("safety_rule", ""),
        is_external_threat=chosen.get("is_external_threat", False),
        comic_archetype=chosen.get("comic_archetype", {}),
    )
