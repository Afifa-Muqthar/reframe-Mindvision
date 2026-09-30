"""
Case Frame: the structured context object capturing the psychological reasoning layer.

Design intent:
Captures user input, emotion analysis, psychological situation parsing,
controllability analysis, strategy selection, and evidence-informed reframing.
Downstream modules (narrative generation, comic storyboard, TTS, UI) read this object.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any


@dataclass
class CaseFrame:
    raw_text: str

    # Filled in by app.nlp.distortions (cognitive distortion scan)
    distortions: list = field(default_factory=list)

    # Filled in by app.emotion.classifier
    emotion_scores: dict = field(default_factory=dict)
    core_emotion: str = ""

    # Psychological Reasoning Layer (app.psychology)
    situation_type: str = ""
    trigger: str = ""
    pattern: str = ""
    is_external_threat: bool = False
    controllable: str = ""
    uncontrollable: str = ""
    strategy: dict = field(default_factory=dict)
    specific_reframe: str = ""
    concrete_action: str = ""
    reasoning: dict = field(default_factory=dict)

    # Summary and techniques
    summary: str = ""
    selected_techniques: list = field(default_factory=list)

    # Longitudinal context
    recurring_theme_note: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def build_summary(self) -> str:
        """Human-readable summary of the psychological assessment."""
        if self.is_external_threat:
            self.summary = (
                f"External interpersonal challenge detected ({self.pattern}); "
                f"focusing on validation, boundaries, and safety rather than internal distortion."
            )
        elif self.pattern:
            self.summary = f"Pattern: {self.pattern}; prioritizing controllable elements and grounded coping."
        else:
            emotion_part = f"feeling {self.core_emotion}" if self.core_emotion else "unclear emotion"
            self.summary = f"{emotion_part}; exploring evidence-informed reframing."
        return self.summary

    def to_dict(self) -> dict:
        return asdict(self)
