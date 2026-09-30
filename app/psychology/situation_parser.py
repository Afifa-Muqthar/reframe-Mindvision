"""
Situation Parser and Psychological Pattern Analyzer.

Classifies incoming thoughts and situations into evidence-informed psychological
patterns and distinguishes internal cognitive distortions from genuine external
threats or unpredictable environments (trauma-informed approach).
"""

import re
from dataclasses import dataclass, asdict
from typing import Optional


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
        "pattern": "Interpersonal threat, volatile environment, normalization/minimization, and subsequent self-doubt.",
        "possible_impact": "Confusion, hypervigilance, self-doubt, and questioning one's own perception because the other person behaves calmly afterward.",
        "is_external_threat": True,
        "controllable": "Your own boundaries, emotional and physical distance, documenting what happened, and seeking trusted support.",
        "uncontrollable": "The other person's volatile behavior, their emotional state, whether they act normal afterward, or trying to fix them.",
    },

    # 2. Performance Anxiety & Catastrophic Prediction
    {
        "situation_id": "performance_anxiety",
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
        "pattern": "Performance anxiety and catastrophic prediction ('I am going to fail').",
        "possible_impact": "Heightened physiological stress, anticipatory dread, and assuming a worst-case outcome before it occurs.",
        "is_external_threat": False,
        "controllable": "Targeted preparation, reviewing key points, taking scheduled rest breaks, and steady breathing.",
        "uncontrollable": "The exact questions that will be asked, the evaluator's moods, or predicting the future with certainty.",
    },

    # 3. Rejection Sensitivity & Mind-Reading
    {
        "situation_id": "rejection_sensitivity",
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
        "pattern": "Mind-reading and rejection sensitivity ('They didn't reply, so everyone must hate me').",
        "possible_impact": "Urge to withdraw socially, frantic checking of devices, and assuming silence equals deliberate hostility.",
        "is_external_threat": False,
        "controllable": "Giving them space, focusing on your own activities, and refraining from catastrophic assumptions.",
        "uncontrollable": "Another person's immediate availability, busy schedule, or phone habits.",
    },

    # 4. Rumination, Regret & Shame Spirals
    {
        "situation_id": "rumination",
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
        "pattern": "Post-event rumination, retroactive regret, and repetitive shame loops.",
        "possible_impact": "Mental exhaustion from running past scenarios in a loop that produces no new resolution.",
        "is_external_threat": False,
        "controllable": "Directing your attention to the present moment, treating yourself with human kindness, and letting the replay go.",
        "uncontrollable": "Changing yesterday's events, words that were already said, or past discomfort.",
    },

    # 5. Harsh Self-Criticism & Overgeneralization
    {
        "situation_id": "self_criticism",
        "regex": [
            r"\bmessed\s+up\s+once\b",
            r"\bi'?m\s+(completely\s+)?(useless|worthless|a\s+loser|broken|an\s+idiot)\b",
            r"\bi\s+can'?t\s+do\s+anything\s+right\b",
            r"\bi\s+always\s+(mess\s+up|ruin\s+everything|fail)\b",
            r"\bi\s+am\s+(useless|worthless|a\s+failure)\b",
        ],
        "trigger": "Encountering a single mistake or perceived flaw in personal performance.",
        "pattern": "Harsh self-criticism, global self-labeling, and overgeneralization ('I messed up once, so I'm useless').",
        "possible_impact": "Deepened shame, loss of confidence, and feeling that one mistake ruins all future prospects.",
        "is_external_threat": False,
        "controllable": "Constructive adjustment, addressing the specific single error, and patient self-talk.",
        "uncontrollable": "Expecting flawless perfection at all times, or never making human errors.",
    },

    # 6. Uncertainty & Threat Anticipation
    {
        "situation_id": "uncertainty",
        "regex": [
            r"\b(don'?t|do\s+not)\s+know\s+what\s+will\s+happen\b",
            r"\bscared\s+(that\s+)?something\s+bad\s+will\s+happen\b",
            r"\bafraid\s+of\s+the\s+future\b",
            r"\bwhat\s+if\s+something\s+(bad|terrible|awful)\s+happens\b",
            r"\buncertain\s+about\s+what\b",
            r"\bdread(ing)?\s+the\s+future\b",
        ],
        "trigger": "Ambiguous future scenario or general lack of clarity on what lies ahead.",
        "pattern": "Threat anticipation, intolerance of uncertainty, and perceived lack of control.",
        "possible_impact": "Constant scanning for danger and treating ambiguity as an inevitable catastrophe.",
        "is_external_threat": False,
        "controllable": "Grounding in the present physical space, managing current routine tasks, and one step at a time.",
        "uncontrollable": "Predicting or controlling future outcomes that are currently outside your agency.",
    },

    # 7. Loss, Breakup & Attachment Distress
    {
        "situation_id": "loss_and_grief",
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
        "pattern": "Loss, acute grief, attachment distress, and longing for past connection.",
        "possible_impact": "Sense of disorientation, waves of sorrow, and self-blame regarding why things ended.",
        "is_external_threat": False,
        "controllable": "Allowing yourself space to grieve without judgment, taking care of basic physical needs, and reaching out to safe friends.",
        "uncontrollable": "The other person's decisions, undoing the past, or rushing the natural timeline of emotional recovery.",
    },

    # 8. Overwhelm & Behavioral Freeze
    {
        "situation_id": "overwhelm",
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
        "pattern": "Task overwhelm, perceived lack of control, and behavioral freeze / paralysis.",
        "possible_impact": "Feeling frozen and guilty, avoiding starting anything because finishing everything seems impossible.",
        "is_external_threat": False,
        "controllable": "Choosing just ONE single 5-minute micro-action, setting a timer, and ignoring the rest temporarily.",
        "uncontrollable": "Finishing all twenty tasks in a single afternoon.",
    },

    # 9. Social Comparison & Inadequacy
    {
        "situation_id": "social_comparison",
        "regex": [
            r"\beveryone\s+is\s+doing\s+better\s+than\s+me\b",
            r"\bdoing\s+better\s+than\s+me\b",
            r"\bfalling\s+behind\s+(everyone|in\s+life)\b",
            r"\bcompared\s+to\s+(everyone|others|them)\b",
            r"\bpeers\s+are\s+(so\s+much\s+)?(more\s+successful|ahead)\b",
            r"\bwhy\s+is\s+everyone\s+else\s+succeeding\b",
        ],
        "trigger": "Observing peers' achievements, timelines, or curated highlights.",
        "pattern": "Social comparison, self-evaluation, and perceived personal inadequacy.",
        "possible_impact": "Feeling like you are failing simply because other people are on different timelines.",
        "is_external_threat": False,
        "controllable": "Identifying your own intrinsic values, respecting your personal timeline, and setting boundaries with social media.",
        "uncontrollable": "Other people's timelines, advantages, or what they choose to share publicly.",
    },

    # 10. Burnout & Physical Exhaustion
    {
        "situation_id": "burnout",
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
    from cognitive distortions.
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
                )

    # General stress fallback (not generic gratitude; empathetic and grounded)
    return PsychologicalProfile(
        raw_text=cleaned,
        situation_id="general_stress",
        category="cognitive",
        trigger="Navigating a challenging personal situation or emotional uncertainty.",
        pattern="Personal stress, ambiguity, and self-doubt regarding the way forward.",
        possible_impact="Feeling weighed down and unsure of what constructive step to take next.",
        is_external_threat=False,
        controllable="Your choices today, treating yourself with patience, and one manageable step.",
        uncontrollable="Fixing everything all at once or controlling past events.",
    )
