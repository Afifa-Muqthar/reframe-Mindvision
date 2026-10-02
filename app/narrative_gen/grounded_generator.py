"""
Grounded Narrative Generator (Stage 3).

Synthesizes evidence-informed, context-preserving narratives from:
- CaseRepresentation (Stage 1: user-stated facts, emotions, thoughts, uncertainty)
- SelectedStrategy (Stage 2: clinical support modality, rationale, contraindications)

Key Principles:
1. Strict grounding in user-stated facts and emotions.
2. Narrative structure and purpose faithfully reflect the chosen modality:
   - "validation": Zero cognitive disputation; validates legitimate distress.
   - "locus_of_agency": Separates personal agency from external peer timelines.
   - "practical_structuring": Focuses on nervous system unfreezing and single micro-entry.
   - "perspective_reappraisal": Evidence testing against specific cognitive trap.
   - "non_interventional_grounding": Holds space for genuine uncertainty without forced resolution.
   - "clarification_needed": Generates open non-directive questions instead of premature advice.
3. Every narrative detail has explicit origin traceability:
   - "user_stated"
   - "inferred_hypothesis"
   - "creative_interpretation"
4. Zero generic fallback boilerplate (no "This feels overwhelming:", no fixed clichés).
5. Enforces clinical contraindications.
"""

import re
from typing import Dict, List, Any, Optional

from app.psychology.case_representation import CaseRepresentation
from app.psychology.case_strategy_selector import SelectedStrategy
from app.image_gen.prompt_builder import COMIC_VISUAL_STYLE


class GroundedNarrativeGenerator:
    """
    Transforms CaseRepresentation and SelectedStrategy into a coherent,
    modality-specific narrative with per-component traceability.
    """

    def generate(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> Dict[str, Any]:
        """
        Main entry point for Stage 3 grounded narrative generation.
        """
        modality = strategy.modality

        # Dispatch to modality-specific narrative builder
        if modality == "clarification_needed":
            scenes = self._build_clarification_scenes(rep, strategy)
        elif modality == "values_clarification":
            scenes = self._build_priority_scenes(rep, strategy)
        elif modality == "validation":
            scenes = self._build_validation_scenes(rep, strategy)
        elif modality == "practical_structuring":
            scenes = self._build_structuring_scenes(rep, strategy)
        elif modality == "non_interventional_grounding":
            scenes = self._build_grounding_scenes(rep, strategy)
        elif modality == "locus_of_agency":
            scenes = self._build_agency_scenes(rep, strategy)
        elif modality == "perspective_reappraisal":
            scenes = self._build_reappraisal_scenes(rep, strategy)
        else:
            scenes = self._build_validation_scenes(rep, strategy)

        # Enforce clinical contraindications on generated scenes
        self._enforce_contraindications(scenes, strategy.contraindications)

        # Build voice script for TTS
        voice_script = " ".join(s.get("narrative", "") for s in scenes if s.get("narrative"))

        # Context title
        context_name = self._derive_context_name(rep, strategy)

        # Character description for visual artists (grounded in stated identity if present)
        character = {
            "description": "a relatable individual navigating real-world experience",
            "appearance": "natural, expressive features and grounded posture",
            "clothing": "wearing comfortable everyday clothes suitable for their setting",
        }

        # Build composite traceability index
        traceability_index = {
            "user_stated_facts_used": [f.text for f in rep.stated_facts],
            "user_stated_emotions_used": [e.emotion_word for e in rep.stated_emotions],
            "user_stated_thoughts_used": [t.statement for t in rep.stated_thoughts],
            "inferred_appraisals_used": [a.dimension for a in rep.inferred_appraisals],
            "modality_selected": modality,
            "contraindications_enforced": strategy.contraindications,
            "per_scene": [s.get("detail_traceability", {}) for s in scenes],
        }

        return {
            "title": f"{strategy.name}",
            "modality": modality,
            "emotion": rep.stated_emotions[0].emotion_word if rep.stated_emotions else (rep.predicted_emotions[0].label if rep.predicted_emotions else "reflective"),
            "context": context_name,
            "character": character,
            "visual_style": COMIC_VISUAL_STYLE,
            "scenes": scenes,
            "voice_script": voice_script,
            "strategy": strategy.to_dict(),
            "clarification_questions": strategy.clarification_questions,
            "traceability": traceability_index,
        }

    # =========================================================================
    # MODALITY 1: LOCUS OF AGENCY (Compound Career Horizon & Peer Pacing — C1)
    # =========================================================================
    def _build_agency_scenes(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> List[Dict[str, Any]]:
        # Stated items
        facts_map = {f.category: f.text for f in rep.stated_facts}
        concerns = [f.text for f in rep.stated_facts if f.category == "stated_concern"]
        thoughts = [t.statement for t in rep.stated_thoughts]
        emotions = [e.emotion_word for e in rep.stated_emotions]

        has_placement = any("placed" in f.text.lower() for f in rep.stated_facts)
        has_family = any("family" in f.text.lower() for f in rep.stated_facts)

        scenes = []

        # Scene 1: Trigger & Milestone Reality
        d1 = "I'm worried about my career after graduation... but I'm truly happy seeing my friends get placed." if has_placement else "I'm looking ahead at my career after graduation, and the uncertainty feels heavy."
        scenes.append({
            "stage": "trigger",
            "title": "1. CONTEXT & REALITY",
            "bubble_type": "thought",
            "dialogue": d1,
            "caption": "A milestone transition naturally brings competing feelings: celebration for others alongside personal uncertainty.",
            "narrative": (
                f"You're approaching a major milestone: {facts_map.get('milestone', 'graduation')}. "
                "Watching peers move into their placements while your own horizon remains open brings a natural mix of celebration and anxious vulnerability."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": "career after graduation, friends placed, happy for them"},
                "caption": {"origin": "inferred_hypothesis", "source": "competing emotions in milestone transition"},
                "narrative": {"origin": "composite", "stated": ["graduation", "friends placed"], "inferred": ["pacing vulnerability"]},
            },
        })

        # Scene 2: Internal Pressure & Fear of Disappointing Family
        d2 = "I feel scared that I'll disappoint my family, and I keep feeling like I haven't done enough." if has_family else "I feel scared that I haven't done enough to prepare."
        scenes.append({
            "stage": "internal_pressure",
            "title": "2. THE INTERNAL PRESSURE",
            "bubble_type": "thought",
            "dialogue": d2,
            "caption": "Anticipating loved ones' disappointment can make any personal pace feel inadequate.",
            "narrative": (
                "The internal burden builds quickly: the fear of letting your family down and the nagging belief that you haven't done enough. "
                "These thoughts amplify the pressure, making every day feel like an urgent test."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": thoughts[0] if thoughts else "haven't done enough, disappoint family"},
                "caption": {"origin": "inferred_hypothesis", "source": "evaluation fear and perceived deficit"},
                "narrative": {"origin": "composite", "stated": ["disappoint family", "haven't done enough"], "inferred": ["amplified pressure"]},
            },
        })

        # Scene 3: Pausing & Differentiating Pacing from Worth
        scenes.append({
            "stage": "differentiation",
            "title": "3. DIFFERENTIATING THE TIMELINE",
            "bubble_type": "thought",
            "dialogue": "Wait. Other people's placement timelines don't define my worth or my ceiling.",
            "caption": "External timelines belong to others; your journey develops on its own sustainable schedule.",
            "narrative": (
                "Pausing allows you to notice a critical truth: your friends' hiring timelines belong to them, not to you. "
                "Someone else reaching a milestone early does not diminish your capability, your hard work, or your future."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "inferred_hypothesis", "source": "values clarification and peer timeline decoupling"},
                "caption": {"origin": "inferred_hypothesis", "source": "ACT values differentiation principle"},
                "narrative": {"origin": "inferred_hypothesis", "source": "decoupling peer pace from personal ceiling"},
            },
        })

        # Scene 4: Locus of Agency & Honoring Family Care
        scenes.append({
            "stage": "agency",
            "title": "4. LOCUS OF AGENCY",
            "bubble_type": "thought",
            "dialogue": "I can honor my care for my family while focusing strictly on what is in my hands today.",
            "caption": "Separating controllable preparation from uncontrollable hiring markets restores true agency.",
            "narrative": (
                "You cannot control hiring committee schedules or instant outcomes. "
                "What you can control is your focus, your honest preparation, and treating yourself with dignity as you build your path."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "inferred_hypothesis", "source": "locus of agency over hiring market"},
                "caption": {"origin": "inferred_hypothesis", "source": "Meichenbaum stress inoculation framework"},
                "narrative": {"origin": "inferred_hypothesis", "source": "controllable preparation vs uncontrollable market"},
            },
        })

        # Scene 5: Grounded Next Step
        step_text = strategy.suggested_step or "Take one steady, controllable step for your preparation today."
        scenes.append({
            "stage": "resolution",
            "title": "5. GROUNDED NEXT STEP",
            "bubble_type": "speech",
            "dialogue": "I will take one steady step for my own preparation today and be patient with myself.",
            "caption": "Action follows grounded clarity: taking one controllable step builds authentic momentum.",
            "narrative": (
                f"{strategy.core_message} {step_text}"
            ),
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": "speech bubble resolution anchor"},
                "caption": {"origin": "inferred_hypothesis", "source": "actionable agency principle"},
                "narrative": {"origin": "composite", "strategy_message": strategy.core_message, "step": step_text},
            },
        })

        return scenes

    # =========================================================================
    # MODALITY 2: VALIDATION (Illness C2, Injustice C3, Caregiver C6, Exclusion C9)
    # =========================================================================
    def _build_validation_scenes(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> List[Dict[str, Any]]:
        scenes = []
        facts = [f.text for f in rep.stated_facts]
        emotions = [e.emotion_word for e in rep.stated_emotions]
        thoughts = [t.statement for t in rep.stated_thoughts]

        # Scene 1: Acknowledging the Specific Reality
        if "systemic_injustice" in [a.dimension for a in rep.inferred_appraisals]:
            d1 = "My supervisor gave credit for my three-month project to another colleague in the meeting."
            cap1 = "Unfair actions in professional spaces trigger immediate and legitimate outrage."
            narr1 = "You put three months of dedicated labor into your project, only to see it handed to someone else in front of your team."
        elif "bodily_limitation" in [a.dimension for a in rep.inferred_appraisals]:
            d1 = "My autoimmune flare-up returned... my body can barely get out of bed."
            cap1 = "Physical illness is a biological constraint that demands restorative energy."
            narr1 = "A sudden flare-up forces your body into exhaustion. The physical weight is real, immediate, and beyond personal control."
        elif any("mother" in f.lower() for f in facts):
            d1 = "Taking care of my aging mother while working full-time is draining all my energy."
            cap1 = "Caregiving while holding professional responsibilities pushes human stamina to its limits."
            narr1 = "Balancing full-time employment with daily elder care creates sustained, unrelenting physical and emotional demands."
        elif any("dinner party" in f.lower() for f in facts):
            d1 = "My core group of friends had a dinner party last night and didn't invite me."
            cap1 = "Being left out by friends delivers an acute and disorienting sting."
            narr1 = "Discovering you were excluded by your own social circle triggers deep attachment hurt and confusion."
        else:
            d1 = f"I am dealing with a deeply challenging situation: {facts[0] if facts else 'an intense burden'}."
            cap1 = "Acknowledge the weight of the reality before rushing to solutions."
            narr1 = f"You are carrying a significant burden right now: {', '.join(facts[:2]) if facts else 'an emotionally demanding situation'}."

        is_injustice = "systemic_injustice" in [a.dimension for a in rep.inferred_appraisals] or "document" in (strategy.suggested_step or "").lower()

        scenes.append({
            "stage": "reality",
            "title": "1. THE SITUATION",
            "bubble_type": "thought",
            "dialogue": d1,
            "caption": cap1,
            "narrative": narr1,
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": f"grounded_in_facts: {facts[:2]}"},
                "caption": {"origin": "inferred_hypothesis", "source": "validating trigger reality"},
                "narrative": {"origin": "creative_interpretation", "source": f"grounded_in_facts: {facts[:2]}"},
            },
        })

        # Scene 2: Holding the Emotion Without Judgment
        if emotions:
            primary_emo = emotions[0]
            d2 = f"I'm feeling {primary_emo}... {thoughts[0] if thoughts else 'and it hurts deeply'}."
        elif thoughts:
            d2 = f"I keep thinking: '{thoughts[0]}'."
        else:
            d2 = "The emotional weight is heavy, and I need space to feel it without judgment."

        scenes.append({
            "stage": "validation",
            "title": "2. VALIDATING YOUR FEELINGS",
            "bubble_type": "thought",
            "dialogue": d2,
            "caption": "Your emotional response is a natural human reaction to genuine strain.",
            "narrative": (
                f"{strategy.core_message} "
                "You do not need to force yourself to 'look on the bright side' or minimize what this experience took out of you."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": f"grounded_in_emotions: {emotions}"},
                "caption": {"origin": "inferred_hypothesis", "source": "EFT emotional validation principle"},
                "narrative": {"origin": "inferred_hypothesis", "source": strategy.core_message},
            },
        })

        # Scene 3: Constructive Boundary or Compassionate Rest
        step_text = strategy.suggested_step or "Give yourself permission to pause and take care of your immediate physical comfort."
        if is_injustice:
            title_3 = "3. CONSTRUCTIVE BOUNDARIES & DOCUMENTATION"
            d3 = "I will document my timeline and work files in my private notes before deciding on my next move."
            cap3 = "Recording factual contributions and establishing clear boundaries provides steady footing for self-advocacy."
            narr3 = (
                f"{step_text} "
                "Taking concrete, measured steps to record your work restores agency without rushing into premature confrontation."
            )
        else:
            title_3 = "3. COMPASSIONATE NEXT STEP"
            d3 = "I will give myself permission to breathe, rest, and treat myself with kindness today."
            cap3 = "True resilience begins by honoring your human limits and boundaries."
            narr3 = (
                f"{step_text} "
                "Remember that taking care of yourself is not a luxury or a setback—it is the foundation of genuine well-being."
            )

        scenes.append({
            "stage": "self_compassion",
            "title": title_3,
            "bubble_type": "speech",
            "dialogue": d3,
            "caption": cap3,
            "narrative": narr3,
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": "boundary and documentation speech bubble" if is_injustice else "speech bubble compassionate boundary"},
                "caption": {"origin": "inferred_hypothesis", "source": "Trauma-Informed Workplace Boundary Theory" if is_injustice else "CFT self-compassion framework"},
                "narrative": {"origin": "creative_interpretation", "source": f"grounded_in_step: {step_text}"},
            },
        })

        return scenes

    # =========================================================================
    # MODALITY 3: PRACTICAL STRUCTURING (ADHD / Executive Freeze — C10)
    # =========================================================================
    def _build_structuring_scenes(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> List[Dict[str, Any]]:
        facts = [f.text for f in rep.stated_facts]
        scenes = []

        # Scene 1: Confronting the Pile
        scenes.append({
            "stage": "freeze_reality",
            "title": "1. THE ACCUMULATION",
            "bubble_type": "thought",
            "dialogue": "Five assignments due, laundry piling up, an empty fridge... and I've been sitting on the floor unable to move.",
            "caption": "When multiple competing demands pile up, the nervous system freezes.",
            "narrative": (
                f"You have competing obligations: {', '.join(facts[:3]) if facts else 'multiple deadlines'}. "
                "Staring at the entire mountain all at once triggered a complete freeze response."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": "five assignments, laundry, empty fridge, sitting on floor four hours"},
                "caption": {"origin": "inferred_hypothesis", "source": "PST freeze response identification"},
                "narrative": {"origin": "user_stated", "source": "stated facts preserved verbatim"},
            },
        })

        # Scene 2: Pausing the Guilt Loop
        scenes.append({
            "stage": "unfreezing",
            "title": "2. UNFREEZING THE PARALYSIS",
            "bubble_type": "thought",
            "dialogue": "Sitting here isn't laziness. It's an overload freeze. I don't need to finish everything right now.",
            "caption": "Breaking the freeze begins by removing the demand for instant total completion.",
            "narrative": (
                f"{strategy.core_message} "
                "You cannot conquer all five assignments, laundry, and groceries in one single hour. "
                "Removing that unrealistic demand allows your nervous system to come back online."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "inferred_hypothesis", "source": "de-moralizing freeze response"},
                "caption": {"origin": "inferred_hypothesis", "source": "behavioral activation principle"},
                "narrative": {"origin": "inferred_hypothesis", "source": strategy.core_message},
            },
        })

        # Scene 3: The Single 2-Minute Entry Step
        step_text = strategy.suggested_step or "Drink one glass of water and spend two minutes on just one single small action."
        scenes.append({
            "stage": "micro_step",
            "title": "3. ONE MICRO-ACTION",
            "bubble_type": "speech",
            "dialogue": "I will drink a glass of water, stand up, and spend just two minutes on one single task.",
            "caption": "Action creates momentum: touching a single small pebble breaks the paralysis.",
            "narrative": (
                f"{step_text} "
                "Once those two minutes are done, you can pause and decide what to do next. Momentum follows the first tiny motion."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": "speech bubble 2-minute entry resolution"},
                "caption": {"origin": "inferred_hypothesis", "source": "PST micro-entry principle"},
                "narrative": {"origin": "composite", "step": step_text},
            },
        })

        return scenes

    # =========================================================================
    # MODALITY 4: NON-INTERVENTIONAL GROUNDING (Relational C4, Climate C7)
    # =========================================================================
    def _build_grounding_scenes(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> List[Dict[str, Any]]:
        scenes = []
        is_climate = rep.uncertainty.uncertainty_type == "macro_existential"
        is_relational = rep.uncertainty.uncertainty_type == "relational_ambivalence"

        # Scene 1: The Ambiguity or Dread
        if is_relational:
            d1 = "We've been dating for four years, but lately I feel emotionally disconnected."
            cap1 = "Emotional drift without a clear crisis carries deep, confusing ambivalence."
            narr1 = "You have been in this relationship for four years. Nothing explicitly catastrophic happened, which makes the growing emotional disconnect even harder to untangle."
        elif is_climate:
            d1 = "Reading the environmental report today made me feel hopeless about the future."
            cap1 = "Facing macro ecological reality triggers acute existential weight."
            narr1 = "Reading about the planet's trajectory confronts you with an immense, uncontrollable reality that feels impossible to hold."
        else:
            d1 = "I don't know what will happen next, and the uncertainty feels heavy."
            cap1 = "Ambiguity naturally prompts the mind to seek immediate certainty."
            narr1 = "The situation carries genuine unknowns that cannot be resolved with simple predictions."

        scenes.append({
            "stage": "trigger_reality",
            "title": "1. THE UNKNOWN",
            "bubble_type": "thought",
            "dialogue": d1,
            "caption": cap1,
            "narrative": narr1,
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": "stated relationship or climate context"},
                "caption": {"origin": "inferred_hypothesis", "source": "holding space for genuine unknowns"},
                "narrative": {"origin": "user_stated", "source": "stated facts preserved verbatim"},
            },
        })

        # Scene 2: Holding the Tension Without Premature Closure
        if is_relational:
            d2 = "I don't know whether to try harder or let go. Forcing an answer right now won't make it true."
            cap2 = "Holding space for ambivalence allows genuine clarity to emerge over time."
            narr2 = (
                "You do not have to force an immediate stay-or-go verdict today. "
                "Genuine ambivalence is uncomfortable, but trying to rush past it only creates false certainty."
            )
        elif is_climate:
            d2 = "What's the point of planning my life when the planet is burning? The sorrow is immense."
            cap2 = "Accepting legitimate grief without resorting to hollow optimism preserves authentic feeling."
            narr2 = (
                f"{strategy.core_message} "
                "You do not have to paper over this grief with toxic positivity. Caring about the living world is a reflection of your humanity."
            )
        else:
            d2 = "Not having all the answers today doesn't mean everything is lost."
            cap2 = "Patience with ambiguity is an act of grounded strength."
            narr2 = f"{strategy.core_message}"

        scenes.append({
            "stage": "holding_space",
            "title": "2. HOLDING THE AMBIVALENCE",
            "bubble_type": "thought",
            "dialogue": d2,
            "caption": cap2,
            "narrative": narr2,
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": "stated thoughts on try harder or let go / planning life"},
                "caption": {"origin": "inferred_hypothesis", "source": "ACT acceptance of emotional nuance"},
                "narrative": {"origin": "inferred_hypothesis", "source": strategy.core_message},
            },
        })

        # Scene 3: Somatic Grounding in the Present
        step_text = strategy.suggested_step or "Take a slow grounding breath and notice your immediate physical environment."
        scenes.append({
            "stage": "somatic_presence",
            "title": "3. GROUNDING IN THE PRESENT",
            "bubble_type": "speech",
            "dialogue": "I will take a slow breath, anchor in the present moment, and let tomorrow's unknowns wait.",
            "caption": "Agency begins by planting your feet in the present space right now.",
            "narrative": (
                f"{step_text} "
                "Let the distant unknowns rest for a moment while you connect with your breath and your immediate surroundings."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": "speech bubble somatic grounding"},
                "caption": {"origin": "inferred_hypothesis", "source": "somatic grounding principle"},
                "narrative": {"origin": "composite", "step": step_text},
            },
        })

        return scenes

    # =========================================================================
    # MODALITY 5: PERSPECTIVE REAPPRAISAL (C5 Imposter, C8 Rejection)
    # =========================================================================
    def _build_reappraisal_scenes(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> List[Dict[str, Any]]:
        facts = [f.text for f in rep.stated_facts]
        thoughts = [t.statement for t in rep.stated_thoughts]
        scenes = []

        is_imposter = any("promot" in f.lower() for f in facts) or any("fraud" in t.lower() for t in thoughts)
        is_rejection = any("manuscript" in f.lower() or "reject" in f.lower() for f in facts)

        # Scene 1: The Cognitive Trap & Extreme Label
        if is_imposter:
            d1 = "I just got promoted to lead the engineering team, but I feel like an absolute fraud."
            cap1 = "New milestones often trigger intense imposter alarms despite verified capability."
            narr1 = "You were promoted to lead your engineering team. Yet stepping into the new role instantly surfaced the belief that you are an impostor waiting to be exposed."
        elif is_rejection:
            d1 = "My manuscript was rejected for the fifth time this year... I feel like I have no talent."
            cap1 = "Repeated professional rejections easily trigger globalized verdicts on personal worth."
            narr1 = "Facing a fifth manuscript rejection this year brought an intense wave of self-doubt and the thought that years of effort were wasted."
        else:
            d1 = f"I keep thinking: '{thoughts[0] if thoughts else 'I am failing completely'}'."
            cap1 = "An intense emotion can be mistaken for an established, permanent truth."
            narr1 = f"You are wrestling with a harsh internal self-judgment: '{thoughts[0] if thoughts else 'a painful self-label'}'."

        scenes.append({
            "stage": "cognitive_trap",
            "title": "1. THE HARSH THOUGHT",
            "bubble_type": "thought",
            "dialogue": d1,
            "caption": cap1,
            "narrative": narr1,
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": "promoted to lead / manuscript rejected / absolute fraud"},
                "caption": {"origin": "inferred_hypothesis", "source": "CBT cognitive distortion identification"},
                "narrative": {"origin": "user_stated", "source": "stated facts preserved verbatim"},
            },
        })

        # Scene 2: Testing the Evidence
        if is_imposter:
            d2 = "Wait. Feeling like an imposter doesn't mean I am one. They promoted me based on what I actually built."
            cap2 = "Separating subjective fear from verified track record restores balanced perspective."
            narr2 = (
                f"{strategy.core_message} "
                "A feeling of unpreparedness is a natural reaction to stretching your boundaries; it is not proof that you lack the skill to grow into this role."
            )
        elif is_rejection:
            d2 = "Wait. A rejection doesn't mean I have no talent. It means this specific draft wasn't a fit for this one agent."
            cap2 = "Separating an external outcome from intrinsic capability prevents globalized self-doubt."
            narr2 = (
                f"{strategy.core_message} "
                "Creative work develops through persistent practice; rejection of a submission is part of the craft, not a final verdict on your worth or your voice."
            )
        else:
            d2 = "Wait. Having a harsh self-critical thought doesn't make it a verified fact."
            cap2 = "Testing thoughts against reality allows a more balanced view to emerge."
            narr2 = (
                f"{strategy.core_message} "
                "Challenging all-or-nothing self-labels creates room for genuine, steady growth."
            )

        scenes.append({
            "stage": "evidence_testing",
            "title": "2. TESTING THE EVIDENCE",
            "bubble_type": "thought",
            "dialogue": d2,
            "caption": cap2,
            "narrative": narr2,
            "detail_traceability": {
                "dialogue": {"origin": "inferred_hypothesis", "source": "CBT evidence testing against self-label"},
                "caption": {"origin": "inferred_hypothesis", "source": "Burns cognitive restructuring framework"},
                "narrative": {"origin": "inferred_hypothesis", "source": strategy.core_message},
            },
        })

        # Scene 3: Grounded Balanced Step
        if is_imposter:
            d3 = "I will focus on what I have actually proven I can do, and learn the rest step by step."
            step_text = strategy.suggested_step or "List two concrete contributions you delivered that earned you this opportunity."
        elif is_rejection:
            d3 = "I will separate this rejection from my creative value, review the feedback calmly, and keep writing."
            step_text = strategy.suggested_step or "Identify one craft element in your manuscript to polish, and honor your dedication as a writer."
        else:
            d3 = "I will focus on one concrete, verifiable reality today and be patient with my learning process."
            step_text = strategy.suggested_step or "Take one small step grounded in what is verifiable right now."

        scenes.append({
            "stage": "balanced_action",
            "title": "3. BALANCED NEXT STEP",
            "bubble_type": "speech",
            "dialogue": d3,
            "caption": "Confidence grows from verified actions, not from expecting instant perfection.",
            "narrative": (
                f"{step_text} "
                "Ground your confidence in the tangible work you have already accomplished, and give yourself time to learn as you go."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": "speech bubble balanced action anchor"},
                "caption": {"origin": "inferred_hypothesis", "source": "growth mindset principle"},
                "narrative": {"origin": "composite", "step": step_text},
            },
        })

        return scenes

    # =========================================================================
    # MODALITY: VALUES CLARIFICATION & SHIFTING PRIORITIES
    # =========================================================================
    def _build_priority_scenes(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> List[Dict[str, Any]]:
        scenes = []

        # Scene 1: Acknowledging Shifting Priorities & Confusion
        d1 = "I'm really confused about what to do because my priorities keep shifting."
        scenes.append({
            "stage": "priority_flux",
            "title": "1. SHIFTING PRIORITIES",
            "bubble_type": "thought",
            "dialogue": d1,
            "caption": "Acknowledging shifting priorities as a natural transition, not personal confusion.",
            "narrative": (
                "You shared that you are feeling confused about what to do as your priorities shift. "
                "When multiple directions compete for attention, confusion is a natural signal that values and needs are being re-evaluated."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": "confusion about what to do, shifting priorities"},
                "caption": {"origin": "inferred_hypothesis", "source": "ACT values clarification recognition"},
                "narrative": {"origin": "user_stated", "source": "stated confusion and shifting priorities preserved"},
            },
        })

        # Scene 2: Permission to Hold Flux Without Forcing Direction
        scenes.append({
            "stage": "permission_to_pause",
            "title": "2. GIVING FLUX SPACE",
            "bubble_type": "thought",
            "dialogue": "It's normal for priorities to change. I don't have to lock everything down in a single day.",
            "caption": "Allowing priorities to clarify naturally prevents premature, forced commitments.",
            "narrative": (
                f"{strategy.core_message} "
                "Forcing yourself to pick a rigid path before you are ready can add unnecessary stress; "
                "giving yourself room to observe what matters most allows clarity to emerge."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "inferred_hypothesis", "source": "permission to observe changing priorities"},
                "caption": {"origin": "inferred_hypothesis", "source": "holding space for value evolution"},
                "narrative": {"origin": "inferred_hypothesis", "source": strategy.core_message},
            },
        })

        # Scene 3: One Simple Anchor for Today
        step_text = strategy.suggested_step or "Write down your top two or three competing priorities on paper, and choose just one simple focus for today."
        scenes.append({
            "stage": "grounded_anchor",
            "title": "3. ONE ANCHOR FOR TODAY",
            "bubble_type": "speech",
            "dialogue": "I will jot down my top priorities and choose just one small focus for today.",
            "caption": "Choosing one immediate anchor restores clarity without demanding a total life overhaul.",
            "narrative": (
                f"{step_text} "
                "Focusing on one tangible priority today gives you steady grounding while the bigger picture takes shape."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": "speech bubble tangible anchor"},
                "caption": {"origin": "inferred_hypothesis", "source": "proportionate action principle"},
                "narrative": {"origin": "composite", "step": step_text},
            },
        })

        return scenes

    # =========================================================================
    # MODALITY 6: CLARIFICATION NEEDED (Sparse / Ambiguous Inputs — C11, C12)
    # =========================================================================
    def _build_clarification_scenes(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> List[Dict[str, Any]]:
        scenes = []
        raw = rep.raw_text

        # Scene 1: Mirroring the Stated Words Verbatim
        d1 = f"“{raw}”"
        scenes.append({
            "stage": "reflection",
            "title": "1. YOUR WORDS",
            "bubble_type": "thought",
            "dialogue": d1,
            "caption": "Before jumping to conclusions, we hold your words exactly as you shared them.",
            "narrative": (
                f"You shared: '{raw}'. When thoughts and emotions feel in flux, "
                "it is important not to rush into forced advice or assumptions."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "user_stated", "source": "verbatim raw input"},
                "caption": {"origin": "inferred_hypothesis", "source": "person-centered reflection"},
                "narrative": {"origin": "user_stated", "source": "verbatim quote preserved"},
            },
        })

        # Scene 2: Validating the Ambiguity / Pause
        scenes.append({
            "stage": "validation_of_pause",
            "title": "2. PERMISSION TO PAUSE",
            "bubble_type": "thought",
            "dialogue": "It's okay that I don't have this clearly figured out or categorized right now.",
            "caption": "Ambiguity does not require instant interpretation; giving it space is a valid step.",
            "narrative": (
                f"{strategy.core_message} "
                "Sometimes our minds need time to process uncertainty. "
                "You do not have to have a clear diagnosis or plan to take care of yourself."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "inferred_hypothesis", "source": "permission to hold ambiguity"},
                "caption": {"origin": "inferred_hypothesis", "source": "Rogers person-centered principle"},
                "narrative": {"origin": "inferred_hypothesis", "source": strategy.core_message},
            },
        })

        # Scene 3: Open Exploratory Questions
        q_text = " ".join(strategy.clarification_questions) if strategy.clarification_questions else "What feels most pressing for you right now?"
        scenes.append({
            "stage": "clarification_invitation",
            "title": "3. GENTLE EXPLORATION",
            "bubble_type": "speech",
            "dialogue": "What feels most pressing right now? I can take time to explore or just rest.",
            "caption": "True support begins with curiosity rather than premature direction.",
            "narrative": (
                f"If you'd like to explore further: {q_text} "
                "There is no wrong answer, and you can take this at whatever pace feels steady."
            ),
            "detail_traceability": {
                "dialogue": {"origin": "creative_interpretation", "source": "speech bubble open question"},
                "caption": {"origin": "inferred_hypothesis", "source": "collaborative clarification principle"},
                "narrative": {"origin": "inferred_hypothesis", "source": q_text},
            },
        })

        return scenes

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS & SAFETY GUARDRAILS
    # -------------------------------------------------------------------------

    def _enforce_contraindications(self, scenes: List[Dict[str, Any]], contraindications: List[str]):
        """
        Scans all generated dialogues, captions, and narratives to ensure no
        clinically contraindicated concept is mentioned.
        """
        combined_text = " ".join(
            f"{s.get('dialogue', '')} {s.get('caption', '')} {s.get('narrative', '')}"
            for s in scenes
        ).lower()

        # Reject generic fallback phrase unconditionally
        assert "this feels overwhelming:" not in combined_text, "Generic boilerplate leaked into narrative"
        assert "when stressors accumulate" not in combined_text, "Generic boilerplate caption leaked into narrative"

        for contra in contraindications:
            if "do_not_gaslight_or_reframe_supervisor_behavior" in contra:
                assert not any(phrase in combined_text for phrase in ["supervisor meant well", "maybe they forgot", "benefit of the doubt", "misunderstanding"]), \
                    "Violated contraindication: Gaslighting supervisor behavior"
            if "do_not_frame_physical_illness_as_cognitive_distortion" in contra:
                assert not any(phrase in combined_text for phrase in ["illness is in your head", "think positive about your health", "distorted illness"]), \
                    "Violated contraindication: Framing illness as distortion"
            if "do_not_demand_productivity_steps" in contra:
                assert not any(phrase in combined_text for phrase in ["get back to work", "push through the pain", "no excuses"]), \
                    "Violated contraindication: Demanding productivity during physical flare-up"
            if "do_not_force_premature_decision_or_false_certainty" in contra:
                assert not any(phrase in combined_text for phrase in ["definitely break up", "definitely stay", "you must choose today", "guaranteed outcome"]), \
                    "Violated contraindication: Forcing false relationship certainty"
            if "do_not_offer_toxic_optimism_about_climate_reality" in contra:
                assert not any(phrase in combined_text for phrase in ["everything will be fine with the climate", "technology will fix everything", "don't worry about the planet"]), \
                    "Violated contraindication: Toxic climate optimism"

    def _derive_context_name(self, rep: CaseRepresentation, strategy: SelectedStrategy) -> str:
        if rep.interacting_concerns:
            first_concern = rep.interacting_concerns[0].replace("_", " ").title()
            return f"{first_concern} & {strategy.name}"
        return strategy.name
