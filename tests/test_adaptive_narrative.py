"""
Test suite for Adaptive Psychological Visual Narrative System.

Verifies:
1. One 3-scene input (simple cognitive challenge).
2. One 4-5 scene input (moderate spiral / overwhelm / social comparison).
3. One 6-8 scene input (complex interpersonal threat / trauma-informed).
4. One case that attempts >8 scenes to confirm strict stopping and clamping (maximum 8).
5. The specific benchmark test:
   "My father lashes out at me and then acts completely normal afterward."
   Confirming that it does NOT produce a simple 3-panel strip.
6. Visual continuity: persistent character anchor across all generated prompts.
7. Storyboard adaptive grid composition for 3, 4, 5, 6, and 8 panels.
"""

import os
import sys
import tempfile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.generator import NarrativeGenerator
from app.narrative_gen.scene_planner import ScenePlanner
from app.image_gen.prompt_builder import build_scene_prompts, build_character_anchor, COMIC_VISUAL_STYLE
from app.image_gen.storyboard import compose_comic_strip
from app.psychology import parse_situation, select_strategy


def test_simple_situation_produces_three_scenes():
    raw_text = "I have an exam tomorrow and I keep thinking I'm going to fail."
    profile = parse_situation(raw_text)
    strategy = select_strategy(profile)
    planner = ScenePlanner()

    case_frame = CaseFrame(
        raw_text=raw_text,
        core_emotion="fear",
        distortions=["catastrophizing"],
        pattern=profile.pattern,
        is_external_threat=profile.is_external_threat,
        strategy=strategy.to_dict(),
    )

    scenes = planner.plan_scenes(case_frame, strategy.to_dict(), strategy.comic_archetype)
    assert len(scenes) == 3
    assert scenes[0]["stage_category"] == "problem"
    assert scenes[1]["stage_category"] == "reframing"
    assert scenes[2]["stage_category"] == "resolution"


def test_moderate_overwhelm_produces_four_to_five_scenes():
    # 4 scenes for multi-task overwhelm paralysis
    raw_text = "I have 20 things to do and can't start."
    profile = parse_situation(raw_text)
    strategy = select_strategy(profile)
    planner = ScenePlanner()

    case_frame = CaseFrame(
        raw_text=raw_text,
        core_emotion="overwhelm",
        distortions=["all_or_nothing"],
        pattern=profile.pattern,
        is_external_threat=profile.is_external_threat,
        strategy=strategy.to_dict(),
    )

    scenes = planner.plan_scenes(case_frame, strategy.to_dict(), strategy.comic_archetype)
    assert 4 <= len(scenes) <= 5
    stages = [s["stage_category"] for s in scenes]
    assert "trigger" in stages or "problem" in stages
    assert "resolution" in stages or "action" in stages


def test_moderate_social_comparison_produces_five_scenes():
    # 5 scenes for cognitive spiral & comparison
    raw_text = "Everyone is doing better than me and I feel completely inadequate."
    profile = parse_situation(raw_text)
    strategy = select_strategy(profile)
    planner = ScenePlanner()

    case_frame = CaseFrame(
        raw_text=raw_text,
        core_emotion="sadness",
        distortions=["comparison", "self_labeling"],
        pattern=profile.pattern,
        is_external_threat=profile.is_external_threat,
        strategy=strategy.to_dict(),
    )

    scenes = planner.plan_scenes(case_frame, strategy.to_dict(), strategy.comic_archetype)
    assert len(scenes) == 5
    stages = [s["stage_category"] for s in scenes]
    assert stages == ["trigger", "spiral", "examination", "reframing", "resolution"]


def test_father_lashing_out_produces_six_scenes_not_three():
    """
    Real test with the benchmark phrase:
    'My father lashes out at me and then acts completely normal afterward.'
    Must confirm that it does NOT produce a simple 3-panel strip.
    """
    raw_text = "My father lashes out at me and then acts completely normal afterward."
    profile = parse_situation(raw_text)
    strategy = select_strategy(profile)
    planner = ScenePlanner()

    case_frame = CaseFrame(
        raw_text=raw_text,
        core_emotion="fear",
        distortions=["personalization"],
        pattern=profile.pattern,
        is_external_threat=profile.is_external_threat,
        strategy=strategy.to_dict(),
    )

    scenes = planner.plan_scenes(case_frame, strategy.to_dict(), strategy.comic_archetype)

    # 1. Confirm it does NOT produce a simple 3-panel strip
    assert len(scenes) != 3, f"Expected adaptive sequence (>3 scenes), got {len(scenes)}"
    assert len(scenes) == 6, f"Expected 6 scenes for trauma-informed threat, got {len(scenes)}"

    # 2. Verify all rich schema fields are present
    for s in scenes:
        assert "scene_id" in s
        assert "stage_category" in s
        assert "narrative_purpose" in s
        assert "psychological_purpose" in s
        assert "psychological_state" in s
        assert "arousal_level" in s["psychological_state"]
        assert "perceived_control" in s["psychological_state"]
        assert "character_state" in s
        assert "character_action" in s["character_state"]
        assert "character_posture" in s["character_state"]
        assert "facial_expression" in s["character_state"]
        assert "visual_staging" in s
        assert "camera_distance" in s["visual_staging"]
        assert "camera_angle" in s["visual_staging"]
        assert "lighting" in s["visual_staging"]
        assert "dialogue" in s
        assert "caption" in s
        assert "narrative" in s

    # 3. Verify trauma-informed stages exist
    stages = [s["stage_category"] for s in scenes]
    assert "incident" in stages
    assert "disorientation" in stages
    assert "validation" in stages
    assert "externalizing" in stages
    assert "boundary" in stages
    assert "support" in stages


def test_rejection_and_clamping_above_eight_scenes():
    """Confirms that candidate progressions attempting > 8 scenes are capped strictly at 8."""
    planner = ScenePlanner()
    raw_text = "Extreme multi-faceted disaster with 15 different cascading problems."
    profile = parse_situation(raw_text)
    strategy = select_strategy(profile)

    case_frame = CaseFrame(
        raw_text=raw_text,
        core_emotion="panic",
        strategy=strategy.to_dict(),
    )

    # Force candidate target count to 12
    candidates = planner._build_moderate_spiral_progression(
        raw_text, "trigger", "reframe", "action", "overwhelm", "panic", {}
    ) * 3  # 15 candidate scenes

    final_scenes = planner._apply_stopping_criteria(candidates, target_count=12)

    assert len(final_scenes) <= 8
    assert len(final_scenes) == 8
    assert final_scenes[-1]["scene_id"] == 8


def test_visual_continuity_preserves_anchor_across_all_panels():
    """Verifies that all scene prompts (3 to 8) share the identical character anchor and art style."""
    char = {
        "description": "a relatable young adult",
        "appearance": "short dark hair, expressive brown eyes",
        "clothing": "wearing a dark gray sweater and dark pants",
    }
    story = {
        "emotion": "confusion",
        "character": char,
        "visual_style": COMIC_VISUAL_STYLE,
        "scenes": [
            {"visual_description": f"scene {i} description", "stage_category": "incident"}
            for i in range(6)
        ]
    }

    prompts = build_scene_prompts(story)
    assert len(prompts) == 6

    anchor = build_character_anchor(char)
    for p in prompts:
        assert anchor in p
        assert COMIC_VISUAL_STYLE in p
        assert "no text, no words" in p


def test_storyboard_adaptive_grid_layout_all_sizes():
    """Verifies that compose_comic_strip handles 3, 4, 5, 6, and 8 panels cleanly without distortion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        for n in [3, 4, 5, 6, 8]:
            panel_paths = []
            scenes = []
            for i in range(n):
                p = os.path.join(tmp_dir, f"panel_{n}_{i+1}.png")
                img = Image.new("RGB", (256, 256), (180 + i * 8, 200, 220))
                img.save(p)
                panel_paths.append(p)

                scenes.append({
                    "stage_category": "reframing",
                    "title": f"{i+1}. SCENE {i+1}",
                    "narrative": f"Narrative for panel {i+1} in a {n}-panel comic.",
                    "dialogue": f"Dialogue {i+1}",
                    "bubble_type": "thought" if i % 2 == 0 else "speech",
                    "caption": f"Insight for scene {i+1}",
                })

            out_path = os.path.join(tmp_dir, f"comic_{n}_panels.png")
            res = compose_comic_strip(panel_paths, scenes, out_path, title=f"TEST COMIC ({n} PANELS)")

            assert os.path.exists(res)
            with Image.open(res) as img:
                w, h = img.size
                if n == 3:
                    assert w > 800 and h < 550
                elif n == 4:
                    assert w >= 550 and h >= 800
                elif n in [5, 6]:
                    assert w >= 800 and h >= 800
                elif n == 8:
                    assert w >= 1050 and h >= 800
