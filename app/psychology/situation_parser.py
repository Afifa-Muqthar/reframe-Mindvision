"""
Situation Parser and Psychological Pattern Analyzer.

Classifies incoming thoughts and situations into evidence-informed psychological
patterns and produces structured reasoning covering:
  1. Situation
  2. Primary emotional experience
  3. Central internal conflict
  4. Automatic thought / question
  5. Relevant evidence-informed psychological pattern
  6. Appropriate coping/reframing strategy
  7. Specific reframe
  8. Concrete next step

Distinguishes genuine environmental threats from internal cognitive distortions
(trauma-informed approach) and preserves uncertainty where the situation is genuinely uncertain.
"""

import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional


@dataclass
class PsychologicalProfile:
    raw_text: str
    situation_id: str
    category: str
    trigger: str
    pattern: str
    possible_impact: str
    is_external_threat: bool
    controllable: str
    uncontrollable: str
    detected_distortion: str = ""

    # The 8 Structured Reasoning Fields:
    situation: str = ""
    primary_emotion: str = ""
    central_conflict: str = ""
    automatic_thought: str = ""
    coping_strategy_name: str = ""
    specific_reframe: str = ""
    concrete_next_step: str = ""

    def __post_init__(self):
        if not self.situation:
            self.situation = self.trigger
        if not self.primary_emotion:
            self.primary_emotion = self.possible_impact

    def structured_reasoning(self) -> Dict[str, str]:
        """Returns the 8 mandatory structured psychological reasoning fields."""
        return {
            "situation": self.situation,
            "primary_emotion": self.primary_emotion,
            "central_conflict": self.central_conflict,
            "automatic_thought": self.automatic_thought,
            "pattern": self.pattern,
            "coping_strategy": self.coping_strategy_name,
            "specific_reframe": self.specific_reframe,
            "concrete_next_step": self.concrete_next_step,
        }

    def to_dict(self) -> dict:
        return asdict(self)


_PATTERNS = [
    # 1. Interpersonal Threat / Abuse / Unpredictable Environment
    {
        "situation_id": "interpersonal_threat",
        "category": "trauma_informed",
        "regex": [
            r"\b(abusive|abuse)\b",
            r"\blash(es|ed|ing)?\s+out\b",
            r"\bact(s|ed|ing)?\s+(all\s+)?normal\b",
            r"\bnormal\s+later\b",
            r"\byell(s|ed|ing)?\s+at\s+me\b",
            r"\bscream(s|ed|ing)?\s+at\s+me\b",
            r"\btoxic\s+(father|mother|parent|boss|partner|relationship)\b",
            r"\bunpredictable\s+(father|mother|parent|anger|rage|behavior|mood)\b",
            r"\bthreaten(s|ed|ing)?\b",
            r"\bgaslight(s|ed|ing)?\b",
            r"\bhits?\s+me\b",
            r"\bscared\s+of\s+(my\s+)?(dad|father|mom|mother|him|her)\b",
            r"\bvolatile\b",
            r"\bwalking\s+on\s+eggshells\b",
        ],
        "trigger": "Unpredictable anger or emotional volatility from a parent or authority figure, followed by normal behavior.",
        "situation": "Parental or authority volatility and unpredictable hostility followed by normal behavior.",
        "primary_emotion": "Shock, hypervigilance, confusion, and acute self-doubt.",
        "central_conflict": "Internalizing responsibility for someone else's volatility and doubting your perception because they act pleasant afterward.",
        "automatic_thought": "They're acting like nothing happened—did I overreact or cause their anger?",
        "pattern": "Interpersonal threat, volatile environment, normalization/minimization, and subsequent self-doubt.",
        "possible_impact": "Confusion, hypervigilance, self-doubt, and questioning one's own perception because the other person behaves calmly afterward.",
        "is_external_threat": True,
        "controllable": "Your own boundaries, emotional and physical distance, documenting what happened, and seeking trusted support.",
        "uncontrollable": "The other person's volatile behavior, their emotional state, whether they act normal afterward, or trying to fix them.",
    },

    # 2. Relationship Ambivalence, Breakup Uncertainty & Separation Grief
    {
        "situation_id": "relationship_ambivalence",
        "category": "cognitive_grief",
        "regex": [
            r"\b(breakup|break\s+up|broke\s+up|ended\s+the\s+relationship)\b.*\b(uncertain(ty)?|wasn'?t\s+sure|not\s+sure|unsure|didn'?t\s+love|did\s+not\s+love|doubt|guilt|regret|mistake)\b",
            r"\b(uncertain(ty)?|wasn'?t\s+sure|not\s+sure|unsure|didn'?t\s+love|did\s+not\s+love|doubt|guilt|regret|mistake)\b.*\b(breakup|break\s+up|broke\s+up|ended\s+the\s+relationship)\b",
            r"\bbreakup\s+because\s+of\s+(my\s+)?uncertainty\b",
            r"\bdid\s+not\s+love\s+him.*painful\b",
            r"\bwasn'?t\s+sure.*love.*breakup\b",
            r"\buncertainty.*did\s+not\s+love\b",
            r"\bnot\s+sure\s+i\s+love(d)?\b",
        ],
        "trigger": "Ending a relationship amid internal uncertainty or ambivalence about romantic feelings.",
        "situation": "Relationship breakup with honest uncertainty about romantic feelings, coupled with separation grief.",
        "primary_emotion": "Grief, ambivalence, guilt, and sorrow after ending a connection.",
        "central_conflict": "Confusing the natural pain of separation with proof that the breakup was a mistake, while wrestling with genuine uncertainty about romantic feelings.",
        "automatic_thought": "If it hurts this much, does that mean I made a mistake or that I actually loved him?",
        "pattern": "Post-breakup ambivalence, emotional reasoning (interpreting grief as proof of error), and separation distress.",
        "possible_impact": "Questioning whether ending the relationship was right simply because separating causes acute emotional pain.",
        "is_external_threat": False,
        "controllable": "Allowing yourself to feel grief without demanding immediate certainty, giving the separation time to settle, and treating yourself with self-compassion.",
        "uncontrollable": "Having 100% certainty right now, forcing feelings of love that weren't there, or avoiding the natural sorrow of an ending.",
    },

    # 3. General Loss, Breakup & Attachment Distress (Without explicit ambivalence)
    {
        "situation_id": "loss_and_grief",
        "category": "social_coping",
        "regex": [
            r"\b(breakup|break\s+up|broke\s+up)\b",
            r"\bex(-boyfriend|-girlfriend)?\b",
            r"\blost\s+my\s+(partner|friend|loved\s+one)\b",
            r"\bdivorce\b",
            r"\bended\s+the\s+relationship\b",
            r"\bheartbroken\b",
            r"\bgrieving\b",
            r"\bcan'?t\s+stop\s+thinking\s+about\s+(my\s+)?(breakup|ex)\b",
        ],
        "trigger": "Relationship ending, separation, or painful emotional loss.",
        "situation": "Relationship ending / separation loss.",
        "primary_emotion": "Heartbreak, acute grief, attachment distress, and longing.",
        "central_conflict": "Struggling to accept the loss of daily companionship and attachment security.",
        "automatic_thought": "I can't imagine life without them; the emptiness is unbearable.",
        "pattern": "Attachment rupture, acute grief, and separation distress.",
        "possible_impact": "Sense of disorientation, waves of sorrow, and self-blame regarding why things ended.",
        "is_external_threat": False,
        "controllable": "Allowing yourself space to grieve without judgment, taking care of basic physical needs, and reaching out to safe friends.",
        "uncontrollable": "The other person's decisions, undoing the past, or rushing the natural timeline of emotional recovery.",
    },

    # 4. Performance Anxiety & Catastrophic Prediction
    {
        "situation_id": "performance_anxiety",
        "category": "cognitive",
        "regex": [
            r"\b(fail|failing|screw\s+up|blow)\s+(my\s+)?(interview|exam|test|presentation|audition|finals|defense)\b",
            r"\bgoing\s+to\s+fail\b",
            r"\binterview\s+tomorrow\b",
            r"\bexam\s+tomorrow\b",
            r"\bfreeze\s+during\s+(the\s+)?(interview|exam|presentation)\b",
            r"\bwon'?t\s+pass\b",
            r"\bperformance\s+anxiety\b",
        ],
        "trigger": "Approaching high-stakes evaluation, exam, or job interview.",
        "situation": "Approaching high-stakes evaluation, exam, or job interview.",
        "primary_emotion": "Performance anxiety, anticipatory dread, and physiological tension.",
        "central_conflict": "Treating anticipatory anxiety as a certain forecast of catastrophic failure.",
        "automatic_thought": "I am going to freeze and fail completely.",
        "pattern": "Performance anxiety and catastrophic prediction ('I am going to fail').",
        "possible_impact": "Heightened physiological stress, anticipatory dread, and assuming a worst-case outcome before it occurs.",
        "is_external_threat": False,
        "controllable": "Targeted preparation, reviewing key points, taking scheduled rest breaks, and steady breathing.",
        "uncontrollable": "The exact questions that will be asked, the evaluator's moods, or predicting the future with certainty.",
    },

    # 5. Rejection Sensitivity & Mind-Reading
    {
        "situation_id": "rejection_sensitivity",
        "category": "cognitive",
        "regex": [
            r"\b(didn'?t|did\s+not)\s+(reply|text\s+back|answer|respond)\b",
            r"\beveryone\s+hates\s+me\b",
            r"\bleft\s+(me\s+)?on\s+read\b",
            r"\bignoring\s+me\b",
            r"\bleft\s+out\b",
            r"\bdoesn'?t\s+like\s+me\b",
            r"\bthey\s+all\s+hate\s+me\b",
            r"\bgetting\s+tired\s+of\s+me\b",
            r"\bnobody\s+wants\s+me\b",
        ],
        "trigger": "Delayed social response, unanswered text message, or perceived interpersonal coldness.",
        "situation": "Unanswered communication, delayed response, or perceived social distance.",
        "primary_emotion": "Vulnerability, fear of exclusion, and social insecurity.",
        "central_conflict": "Filling communicative silence with catastrophic assumptions of total rejection.",
        "automatic_thought": "They haven't replied, so they must hate me and want me gone.",
        "pattern": "Mind-reading and rejection sensitivity ('They didn't reply, so everyone must hate me').",
        "possible_impact": "Urge to withdraw socially, frantic checking of devices, and assuming silence equals deliberate hostility.",
        "is_external_threat": False,
        "controllable": "Giving them space, focusing on your own activities, and refraining from catastrophic assumptions.",
        "uncontrollable": "Another person's immediate availability, busy schedule, or phone habits.",
    },

    # 6. Rumination, Regret & Shame Spirals
    {
        "situation_id": "rumination",
        "category": "cognitive",
        "regex": [
            r"\b(keep|can'?t\s+stop)\s+(thinking|replaying|dwelling)\s+about\s+what\s+happened\b",
            r"\breplaying\s+(an\s+)?embarrassing\b",
            r"\breplaying\s+(it|the\s+moment|yesterday)\b",
            r"\bwhat\s+happened\s+yesterday\b",
            r"\bwhy\s+did\s+i\s+say\s+that\b",
            r"\bawkward\s+thing\s+i\s+said\b",
            r"\bcringing\s+at\b",
            r"\bkeep\s+replaying\b",
        ],
        "trigger": "Recalling a past awkward moment, mistake, or uncomfortable social interaction.",
        "situation": "Dwelling on a past uncomfortable interaction, mistake, or conversation.",
        "primary_emotion": "Post-event regret, shame, and cognitive exhaustion.",
        "central_conflict": "Repetitively looping past events in the false belief that replaying them will change the outcome.",
        "automatic_thought": "Why did I say that? I can't stop replaying how awkward I was.",
        "pattern": "Post-event rumination, retroactive regret, and repetitive shame loops.",
        "possible_impact": "Mental exhaustion from running past scenarios in a loop that produces no new resolution.",
        "is_external_threat": False,
        "controllable": "Directing your attention to the present moment, treating yourself with human kindness, and letting the replay go.",
        "uncontrollable": "Changing yesterday's events, words that were already said, or past discomfort.",
    },

    # 7. Harsh Self-Criticism & Overgeneralization
    {
        "situation_id": "self_criticism",
        "category": "cognitive",
        "regex": [
            r"\bmessed\s+up\s+once\b",
            r"\bi'?m\s+(completely\s+)?(useless|worthless|a\s+loser|broken|an\s+idiot)\b",
            r"\bi\s+can'?t\s+do\s+anything\s+right\b",
            r"\bi\s+always\s+(mess\s+up|ruin\s+everything|fail)\b",
            r"\bi\s+am\s+(useless|worthless|a\s+failure)\b",
        ],
        "trigger": "Encountering a single mistake or perceived flaw in personal performance.",
        "situation": "Encountering a personal mistake, setback, or perceived flaw.",
        "primary_emotion": "Harsh self-blame, inadequacy, and demoralization.",
        "central_conflict": "Equating a single specific mistake with total personal worthlessness.",
        "automatic_thought": "I messed up once, so I am completely useless.",
        "pattern": "Harsh self-criticism, global self-labeling, and overgeneralization ('I messed up once, so I'm useless').",
        "possible_impact": "Deepened shame, loss of confidence, and feeling that one mistake ruins all future prospects.",
        "is_external_threat": False,
        "controllable": "Constructive adjustment, addressing the specific single error, and patient self-talk.",
        "uncontrollable": "Expecting flawless perfection at all times, or never making human errors.",
    },

    # 8. Uncertainty & Threat Anticipation (Non-breakup future ambiguity)
    {
        "situation_id": "uncertainty",
        "category": "cognitive",
        "regex": [
            r"\b(don'?t|do\s+not)\s+know\s+what\s+will\s+happen\b",
            r"\bscared\s+(that\s+)?something\s+bad\s+will\s+happen\b",
            r"\bafraid\s+of\s+the\s+future\b",
            r"\bwhat\s+if\s+something\s+(bad|terrible|awful)\s+happens\b",
            r"\buncertain\s+about\s+(what|the\s+future|my\s+future|career|life)\b",
            r"\bdread(ing)?\s+the\s+future\b",
            r"\b(constant|future)\s+uncertainty\b",
            r"\buncertainty\s+is\s+making\s+me\b",
            r"\buncertainty\s+about\s+(my\s+)?(future|career|path)\b",
        ],
        "trigger": "Ambiguous future scenario or general lack of clarity on what lies ahead.",
        "situation": "Ambiguous future scenario or general lack of predictability.",
        "primary_emotion": "Threat anticipation, dread of the unknown, and vulnerability.",
        "central_conflict": "Demanding absolute guarantees from an inherently unpredictable future.",
        "automatic_thought": "Something terrible is going to happen and I won't be able to handle it.",
        "pattern": "Threat anticipation, intolerance of uncertainty, and perceived lack of control.",
        "possible_impact": "Constant scanning for danger and treating ambiguity as an inevitable catastrophe.",
        "is_external_threat": False,
        "controllable": "Grounding in the present physical space, managing current routine tasks, and one step at a time.",
        "uncontrollable": "Predicting or controlling future outcomes that are currently outside your agency.",
    },

    # 9. Overwhelm & Behavioral Freeze
    {
        "situation_id": "overwhelm",
        "category": "problem_focused",
        "regex": [
            r"\b\d+\s+things\s+to\s+do\b",
            r"\btwenty\s+things\s+to\s+do\b",
            r"\b(can'?t|cannot)\s+start\b",
            r"\bparalyzed\s+by\b",
            r"\bso\s+much\s+to\s+do\s+and\s+(i\s+)?can'?t\b",
            r"\bexecutive\s+dysfunction\b",
            r"\bparalysis\b",
            r"\bto-do\s+list\s+is\s+too\s+long\b",
            r"\bmountain\s+of\s+work\b",
        ],
        "trigger": "Cognitive overload from a long list of responsibilities and deadlines.",
        "situation": "Cognitive overload from multiple competing tasks and deadlines.",
        "primary_emotion": "Executive overload, paralysis, and guilt over inaction.",
        "central_conflict": "Believing you must complete every obligation simultaneously before starting anything.",
        "automatic_thought": "I have twenty things to do and I'm paralyzed, so I can't start.",
        "pattern": "Task overwhelm, perceived lack of control, and behavioral freeze / paralysis.",
        "possible_impact": "Feeling frozen and guilty, avoiding starting anything because finishing everything seems impossible.",
        "is_external_threat": False,
        "controllable": "Choosing just ONE single 5-minute micro-action, setting a timer, and ignoring the rest temporarily.",
        "uncontrollable": "Finishing all twenty tasks in a single afternoon.",
    },

    # 10. Social Comparison & Inadequacy
    {
        "situation_id": "social_comparison",
        "category": "cognitive",
        "regex": [
            r"\beveryone\s+(my\s+age\s+|else\s+)?(is|seems\s+to\s+be)\s+(succeeding|doing\s+(so\s+much\s+|far\s+|way\s+)?better)\b",
            r"\bdoing\s+(so\s+much\s+|far\s+|way\s+)?better\s+than\s+me\b",
            r"\bfalling\s+behind(\s+(everyone|others|in\s+life|peers))?\b",
            r"\bcompared\s+to\s+(everyone|others|them|peers)\b",
            r"\bpeers\s+are\s+(so\s+much\s+)?(more\s+successful|ahead)\b",
            r"\b(why\s+is\s+)?everyone\s+(else\s+)?(is\s+)?succeeding\b",
            r"\bsucceeding\s+while\s+i\s+feel\b",
            r"\bcompare\s+myself\s+to\b",
            r"\bhow\s+to\s+catch\s+up\b",
        ],
        "trigger": "Observing peers' achievements, timelines, or curated highlights.",
        "situation": "Observing peers' milestones, timelines, or curated achievements.",
        "primary_emotion": "Inferiority, envy, self-judgment, and anxiety about falling behind.",
        "central_conflict": "Measuring your internal private struggles against other people's curated public highlights.",
        "automatic_thought": "Everyone is doing better than me; I am failing in life.",
        "pattern": "Social comparison, self-evaluation, and perceived personal inadequacy.",
        "possible_impact": "Feeling like you are failing simply because other people are on different timelines.",
        "is_external_threat": False,
        "controllable": "Identifying your own intrinsic values, respecting your personal timeline, and setting boundaries with social media.",
        "uncontrollable": "Other people's timelines, advantages, or what they choose to share publicly.",
    },

    # 11. Burnout & Physical Exhaustion
    {
        "situation_id": "burnout",
        "category": "biological_pacing",
        "regex": [
            r"\b(exhausted|exhaustion|burnout|burned\s+out)\b",
            r"\bcan'?t\s+keep\s+pushing\b",
            r"\bcompletely\s+drained\b",
            r"\bbody\s+is\s+giving\s+up\b",
            r"\bno\s+energy\s+left\b",
            r"\bchronic\s+fatigue\b",
            r"\brun\s+on\s+empty\b",
        ],
        "trigger": "Prolonged physical strain, chronic work pressure, or depleted sleep reserves.",
        "situation": "Prolonged physical exhaustion and nervous system depletion.",
        "primary_emotion": "Exhaustion, emotional flatness, and physical depletion.",
        "central_conflict": "Believing you are not allowed to rest until you completely collapse.",
        "automatic_thought": "I have no energy left, but if I stop I am lazy.",
        "pattern": "Biological exhaustion, nervous system strain, and misplaced guilt regarding resting.",
        "possible_impact": "Physical depletion, emotional numbness, and cognitive fatigue.",
        "is_external_threat": False,
        "controllable": "Allowing yourself guilt-free restorative rest, drinking water, and stepping away from pressure.",
        "uncontrollable": "Expecting your body to operate indefinitely without recovery.",
    },
]


def parse_situation(text: str) -> PsychologicalProfile:
    """
    Parses user input text to identify the underlying trigger, pattern,
    and controllability profile. Distinguishes genuine environmental threats
    from cognitive distortions and extracts the 8 structured reasoning fields.
    """
    cleaned = (text or "").strip()
    lowered = cleaned.lower()

    for p in _PATTERNS:
        for r in p["regex"]:
            if re.search(r, lowered):
                return PsychologicalProfile(
                    raw_text=cleaned,
                    situation_id=p["situation_id"],
                    category=p.get("category", "cognitive"),
                    trigger=p["trigger"],
                    pattern=p["pattern"],
                    possible_impact=p["possible_impact"],
                    is_external_threat=p["is_external_threat"],
                    controllable=p["controllable"],
                    uncontrollable=p["uncontrollable"],
                    situation=p.get("situation", p["trigger"]),
                    primary_emotion=p.get("primary_emotion", p["possible_impact"]),
                    central_conflict=p.get("central_conflict", ""),
                    automatic_thought=p.get("automatic_thought", ""),
                )

    # General stress fallback (not generic gratitude; empathetic, non-evaluative, and grounded)
    return PsychologicalProfile(
        raw_text=cleaned,
        situation_id="general_stress",
        category="cognitive",
        trigger="Navigating a challenging personal situation or emotional uncertainty.",
        situation="Challenging personal dilemma or emotional uncertainty.",
        primary_emotion="Stress, ambivalence, and emotional vulnerability.",
        central_conflict="Feeling burdened by ambiguity and wanting clarity before moving forward.",
        automatic_thought="I don't know what to do and everything feels heavy.",
        pattern="Personal stress, ambiguity, and self-doubt regarding the way forward.",
        possible_impact="Feeling weighed down and unsure of what constructive step to take next.",
        is_external_threat=False,
        controllable="Your choices today, treating yourself with patience, and one manageable step.",
        uncontrollable="Fixing everything all at once or controlling past events.",
    )
