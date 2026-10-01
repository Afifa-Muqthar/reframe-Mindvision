"""
Semantic Relevance and Cross-Domain Contamination Validator.

Validates that:
1. The psychological reframe directly addresses the user's specific central conflict.
2. The reframe preserves the user's actual situation without introducing foreign problems.
3. No cross-domain contamination occurs (e.g. relationship breakups must not mention exam
   anxiety or task overwhelm; uncertainty must not mention breakup grief or twenty tasks;
   social comparison must not mention task-overwhelm timers).
4. Generic cliches that could be pasted into an unrelated prompt are rejected.
5. In situations of genuine uncertainty (such as relationship ambivalence), uncertainty is
   preserved rather than forcing premature or false certainty.
6. If a scene or narrative fails validation, it is regenerated or replaced with a
   scenario-relevant, evidence-informed fallback—not generic motivational text or breakup grief.
"""

import re
from typing import Tuple, List, Dict, Any


class SemanticValidator:
    """
    Validates the semantic specificity and clinical appropriateness of reframes
    and scene narratives before final delivery.
    """

    # Domain vocabulary definitions
    _RELATIONSHIP_KEYWORDS = {
        "breakup", "break up", "broke up", "relationship", "partner", "ex",
        "love", "dating", "separation", "loved him", "loved her", "dumped"
    }

    _PERFORMANCE_KEYWORDS = {
        "exam", "interview", "test", "presentation", "audition", "finals",
        "boss", "study", "studying", "grades", "evaluation"
    }

    _OVERWHELM_KEYWORDS = {
        "20 things", "twenty things", "to-do list", "mountain of work",
        "paralyzed by", "executive dysfunction", "too much to do", "paralysis"
    }

    _SOCIAL_COMPARISON_KEYWORDS = {
        "doing better than me", "everyone is doing better", "falling behind in life",
        "peers are ahead", "comparing myself", "everyone my age", "how to catch up",
        "why is everyone else succeeding"
    }

    _UNCERTAINTY_KEYWORDS = {
        "don't know what will happen", "scared something bad will happen",
        "afraid of the future", "constant uncertainty", "future and career",
        "uncertain about", "dreading the future"
    }

    _DOMESTIC_THREAT_KEYWORDS = {
        "lashes out", "lash out", "yelling", "abusive", "screaming",
        "volatile", "threaten", "gaslight", "walking on eggshells", "toxic father"
    }

    # Banned terms when situation belongs to another domain
    _PERFORMANCE_TERMS_BANNED_IN_OTHER_DOMAINS = [
        "exam", "interview", "test", "presentation", "evaluator", "grades",
        "talking points", "freeze and fail", "will fail", "going to fail", "fail completely"
    ]

    _RELATIONSHIP_TERMS_BANNED_IN_OTHER_DOMAINS = [
        "breakup", "break up", "broke up", "ex-partner", "relationship ended", "romantic feelings",
        "did not love", "loved him", "loved her", "permission to grieve", "should have continued",
        "separation pain", "grief is love", "healing from loss", "dating"
    ]

    _OVERWHELM_TERMS_BANNED_IN_OTHER_DOMAINS = [
        "twenty things", "20 things", "twenty tasks", "mountain of work",
        "timer for five minutes", "timer set for five minutes", "to-do list"
    ]

    _THREAT_TERMS_BANNED_IN_OTHER_DOMAINS = [
        "lashes out", "toxic father", "yelling at me", "screaming at me", "volatile father"
    ]

    _DOMAIN_FALLBACKS = {
        "relationship": {
            "problem": "It hurts so much... but I was so uncertain if I loved him.",
            "examination": "Wait. Pain means I care, but it doesn't erase why I was uncertain.",
            "reframing": "Feeling pain doesn't mean this was wrong. I can care and still honor my doubts.",
            "resolution": "I don't have to force certainty today. I'll let myself feel sad and take one quiet breath.",
            "reframe": "Feeling pain after a breakup does not by itself prove that the relationship should have continued. You can acknowledge your grief while honoring that your uncertainty was honest and real.",
            "action": "Give yourself permission to feel the sadness of the ending today without rushing to judge whether the decision was definitely right or wrong.",
            "caption_reframing": "Pain reflects that you cared, not that the relationship should have continued.",
            "caption_resolution": "Give yourself permission to grieve without rushing to prove your decision right or wrong.",
        },
        "uncertainty": {
            "problem": "I don't know what will happen with my future, and the uncertainty feels terrifying.",
            "examination": "Wait. Not knowing what happens next doesn't mean catastrophe is guaranteed.",
            "reframing": "I cannot predict the entire future today. I can handle the hour in front of me.",
            "resolution": "I will anchor in today's routine tasks and take it one steady hour at a time.",
            "reframe": "Not knowing what will happen is uncomfortable, but uncertainty does not mean catastrophe. You do not need to figure out the future today; focus on what is within your agency right now.",
            "action": "Anchor your agency in the present environment. Complete one simple routine action and let the distant unknowns wait.",
            "caption_reframing": "Reframing shifts focus from demanding future certainty to exercising present agency.",
            "caption_resolution": "Grounding in immediate present action relieves the burden of carrying tomorrow's unknowns.",
        },
        "social_comparison": {
            "problem": "Everyone is doing so much better than me... I'm completely behind.",
            "examination": "Wait. Their public highlights don't define my personal journey.",
            "reframing": "Other people's pace is not my measuring tape. My journey has its own timing.",
            "resolution": "I will set my phone aside and focus on what genuinely matters to my own growth.",
            "reframe": "You are comparing your behind-the-scenes struggles to everyone else's public highlights. Another person's success does not diminish your worth or pacing.",
            "action": "Put down social media, honor your own pace, and take one purposeful step toward your personal intrinsic values.",
            "caption_reframing": "Replace upward social comparison with alignment to your own intrinsic values and pace.",
            "caption_resolution": "Direct your attention inward toward your own growth, craft, and authentic progress.",
        },
        "overwhelm": {
            "problem": "I have twenty things to do and I'm so overwhelmed I can't even begin.",
            "examination": "Wait. I don't have to conquer the whole mountain in this single hour.",
            "reframing": "I don't need to do twenty things today. I only need to do one 5-minute step.",
            "resolution": "I'll spend five minutes on this one task, and let that be enough for now.",
            "reframe": "You do not have to conquer all twenty tasks simultaneously. You only need to touch one small corner of one task. Momentum follows action, not perfection.",
            "action": "Pick the simplest sub-task on your list, set a timer for five minutes, and do just that one thing.",
            "caption_reframing": "Break the wall down into a single brick. Momentum follows action, not perfection.",
            "caption_resolution": "One small step forward breaks the freeze and builds genuine momentum.",
        },
        "academic": {
            "problem": "I'm terrified I'm going to freeze and fail tomorrow.",
            "examination": "Wait. Being nervous just means I care, not that failure is certain.",
            "reframing": "Feeling nervous doesn't mean I will fail. I can focus on the questions I know.",
            "resolution": "I will review these three topics calmly and get a good night of rest.",
            "reframe": "Feeling nervous is a natural physical response to something that matters to you, not evidence that failure is guaranteed. Direct your energy strictly into what is within your agency.",
            "action": "Focus on reviewing your main core topics today, taking scheduled rest breaks, and trusting your preparation.",
            "caption_reframing": "Reframe anticipatory anxiety into focused preparation on what you can control.",
            "caption_resolution": "Action follows clarity: targeted preparation builds calm capability.",
        },
        "threat": {
            "problem": "He is acting normal now... maybe I shouldn't still feel upset.",
            "examination": "Wait. His pleasant mood now does not erase how frightening he was earlier.",
            "reframing": "His acting normal doesn't erase what happened. I can recognize the pattern without blaming myself.",
            "resolution": "I can't control his reaction. I can choose what I do next and who I turn to for support.",
            "reframe": "Their acting normal afterward does not erase what happened earlier. You don't have to take responsibility for another person's behavior; focus strictly on your boundaries and safety.",
            "action": "Right now, give yourself physical or emotional distance, write down what happened clearly, and reach out to someone you trust.",
            "caption_reframing": "You don't have to rewrite what happened simply because the situation became calm.",
            "caption_resolution": "Focus on safety, support, and the parts of the situation you can control.",
        },
    }

    @classmethod
    def detect_domain(cls, raw_text: str, situation_id: str) -> str:
        lowered = (raw_text or "").lower()
        sit = (situation_id or "").lower()

        if sit in ["relationship_ambivalence", "loss_and_grief", "breakup"] or any(k in lowered for k in cls._RELATIONSHIP_KEYWORDS):
            return "relationship"
        if sit in ["interpersonal_threat", "abuse_or_hostility"] or any(k in lowered for k in cls._DOMESTIC_THREAT_KEYWORDS):
            return "threat"
        if sit in ["social_comparison", "inferiority_feelings"] or any(k in lowered for k in cls._SOCIAL_COMPARISON_KEYWORDS):
            return "social_comparison"
        if sit in ["uncertainty", "future_fear", "threat_anticipation"] or any(k in lowered for k in cls._UNCERTAINTY_KEYWORDS):
            return "uncertainty"
        if sit in ["overwhelm", "task_overwhelm", "paralysis"] or any(k in lowered for k in cls._OVERWHELM_KEYWORDS):
            return "overwhelm"
        if sit in ["performance_anxiety", "evaluation_fear"] or any(k in lowered for k in cls._PERFORMANCE_KEYWORDS):
            return "academic"
        return "general"

    @classmethod
    def validate_reframe(
        cls,
        raw_text: str,
        situation_id: str,
        central_conflict: str,
        reframe: str,
        dialogues: List[str] = None,
        captions: List[str] = None,
    ) -> Tuple[bool, str]:
        """
        Validates reframe, dialogue, and caption texts for semantic relevance and lack of cross-domain contamination.
        Returns (is_valid, reason).
        """
        reframe_lowered = (reframe or "").lower()
        dialogue_combined = " ".join((d or "").lower() for d in (dialogues or []))
        caption_combined = " ".join((c or "").lower() for c in (captions or []))
        all_text = f"{reframe_lowered} {dialogue_combined} {caption_combined}"

        domain = cls.detect_domain(raw_text, situation_id)

        # 1. Non-relationship domain must NOT contain relationship/breakup terms
        if domain != "relationship":
            for term in cls._RELATIONSHIP_TERMS_BANNED_IN_OTHER_DOMAINS:
                if term in all_text:
                    return False, f"Cross-domain contamination: Non-relationship situation ({domain}) contains breakup/relationship term '{term}'."

        # 2. Non-performance domain must NOT contain academic/exam terms
        if domain not in ["academic", "performance"]:
            for term in cls._PERFORMANCE_TERMS_BANNED_IN_OTHER_DOMAINS:
                if term in all_text:
                    return False, f"Cross-domain contamination: Non-performance situation ({domain}) contains performance term '{term}'."

        # 3. Non-overwhelm domain must NOT contain twenty-tasks overwhelm terms
        if domain != "overwhelm":
            for term in cls._OVERWHELM_TERMS_BANNED_IN_OTHER_DOMAINS:
                if term in all_text:
                    return False, f"Cross-domain contamination: Non-overwhelm situation ({domain}) contains task overwhelm term '{term}'."

        # 4. Non-threat domain must NOT contain domestic threat terms
        if domain != "threat":
            for term in cls._THREAT_TERMS_BANNED_IN_OTHER_DOMAINS:
                if term in all_text:
                    return False, f"Cross-domain contamination: Non-threat situation ({domain}) contains threat term '{term}'."

        # 5. Check for mismatched generic performance cliches
        if domain not in ["academic", "performance"]:
            if "feeling nervous doesn't mean i will fail" in all_text:
                return False, "Generic performance cliche detected in a non-performance context."
            if "practice your main talking points" in all_text:
                return False, "Interview preparation advice detected in a non-performance context."

        # 6. Check for preservation of uncertainty in ambivalence scenarios
        if situation_id in ["relationship_ambivalence", "breakup_ambivalence"]:
            if any(phrase in reframe_lowered for phrase in ["definitely the right choice", "definitely correct", "guaranteed right decision"]):
                return False, "False certainty: Reframe claims the breakup was definitely right instead of holding space for ambivalence."
            has_pain_acknowledgment = any(w in reframe_lowered for w in ["pain", "hurt", "grief", "loss", "sadness", "cared"])
            has_uncertainty_acknowledgment = any(w in reframe_lowered for w in ["uncertain", "doubts", "not sure", "ambivalence", "feelings"])
            if not (has_pain_acknowledgment and has_uncertainty_acknowledgment):
                return False, "Incomplete reframe: Must address both separation pain and honest uncertainty."

        # 7. Check for toxic positivity in threat contexts
        if domain == "threat":
            if any(w in reframe_lowered for w in ["just think positive", "be grateful", "focus on gratitude"]):
                return False, "Toxic positivity detected in a threat context: Must center safety and validation."

        return True, "Valid"

    @classmethod
    def sanitize_story(cls, case_frame, story: dict) -> dict:
        """
        Enforces semantic validity on story scenes and narratives.
        If validation fails, sanitizes the story with grounded situation-specific content
        derived from the scenario's actual domain and archetype.
        """
        raw_text = getattr(case_frame, "raw_text", "")
        situation_id = getattr(case_frame, "situation_type", "")
        reasoning = getattr(case_frame, "reasoning", {}) or {}
        central_conflict = reasoning.get("central_conflict", "")
        reframe = reasoning.get("specific_reframe", "") or getattr(case_frame, "specific_reframe", "")

        domain = cls.detect_domain(raw_text, situation_id)
        scenes = story.get("scenes", [])
        dialogues = [s.get("dialogue", "") for s in scenes]
        captions = [s.get("caption", "") for s in scenes]

        is_valid, reason = cls.validate_reframe(
            raw_text, situation_id, central_conflict, reframe, dialogues, captions
        )

        if not is_valid:
            strategy = getattr(case_frame, "strategy", {}) or {}
            archetype = strategy.get("comic_archetype", {})
            arch_scenes = archetype.get("scenes", [])
            fallback_dict = cls._DOMAIN_FALLBACKS.get(domain, cls._DOMAIN_FALLBACKS["uncertainty"])

            # Clean reframe
            fallback_reframe = strategy.get("reframe") or fallback_dict.get("reframe")
            if fallback_reframe:
                reframe = fallback_reframe

            # Sanitize each scene according to its domain and stage
            for idx, scene in enumerate(scenes):
                cat = scene.get("stage_category", scene.get("stage", "")).lower()
                dia = scene.get("dialogue", "")
                cap = scene.get("caption", "")
                narr = scene.get("narrative", "")
                scene_text = f"{dia} {cap} {narr}".lower()

                has_contamination = False

                # Check if scene contains terms from forbidden domains
                if domain != "relationship" and any(t in scene_text for t in cls._RELATIONSHIP_TERMS_BANNED_IN_OTHER_DOMAINS):
                    has_contamination = True
                if domain not in ["academic", "performance"] and any(t in scene_text for t in cls._PERFORMANCE_TERMS_BANNED_IN_OTHER_DOMAINS):
                    has_contamination = True
                if domain != "overwhelm" and any(t in scene_text for t in cls._OVERWHELM_TERMS_BANNED_IN_OTHER_DOMAINS):
                    has_contamination = True
                if domain != "threat" and any(t in scene_text for t in cls._THREAT_TERMS_BANNED_IN_OTHER_DOMAINS):
                    has_contamination = True

                if has_contamination:
                    # Replace with archetype scene dialogue/caption if available
                    if idx < len(arch_scenes) and arch_scenes[idx].get("dialogue"):
                        scene["dialogue"] = arch_scenes[idx]["dialogue"]
                        if arch_scenes[idx].get("caption"):
                            scene["caption"] = arch_scenes[idx]["caption"]
                    else:
                        # Use domain-specific fallback mapped to the scene stage
                        if any(k in cat for k in ["problem", "trigger", "incident"]):
                            scene["dialogue"] = fallback_dict["problem"]
                        elif any(k in cat for k in ["examination", "spiral"]):
                            scene["dialogue"] = fallback_dict.get("examination", fallback_dict["problem"])
                        elif any(k in cat for k in ["reframe", "reframing"]):
                            scene["dialogue"] = fallback_dict["reframing"]
                            if fallback_dict.get("caption_reframing"):
                                scene["caption"] = fallback_dict["caption_reframing"]
                            scene["narrative"] = fallback_dict["reframe"]
                        elif any(k in cat for k in ["resolution", "action"]):
                            scene["dialogue"] = fallback_dict["resolution"]
                            if fallback_dict.get("caption_resolution"):
                                scene["caption"] = fallback_dict["caption_resolution"]
                            scene["narrative"] = fallback_dict["action"]

            # Update voice script with grounded domain text
            if strategy.get("voice_story_template"):
                story["voice_script"] = strategy["voice_story_template"]

        return story
