import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.image_gen.prompt_builder import (
    COMIC_VISUAL_STYLE,
    NEGATIVE_PROMPT,
    build_character_anchor,
    build_scene_prompts,
)


def test_character_anchor_generation():
    char = {
        "description": "a young student in their early 20s",
        "appearance": "short dark tousled hair, expressive brown eyes",
        "clothing": "charcoal gray hoodie and dark jeans",
    }
    anchor = build_character_anchor(char)
    assert "short dark tousled hair" in anchor
    assert "charcoal gray hoodie" in anchor


def test_build_scene_prompts_preserves_anchor_across_all_scenes():
    story = {
        "emotion": "fear",
        "context": "Exam Preparation",
        "character": {
            "description": "a dedicated college student",
            "appearance": "short dark tousled hair, expressive brown eyes",
            "clothing": "charcoal gray hoodie and dark denim jeans",
        },
        "visual_style": COMIC_VISUAL_STYLE,
        "scenes": [
            {
                "stage": "problem",
                "visual_description": "sitting overwhelmed at cluttered desk with open textbooks",
            },
            {
                "stage": "reframing",
                "visual_description": "sitting at desk taking a deep breath looking at notes",
            },
            {
                "stage": "resolution",
                "visual_description": "sitting upright writing a revision checklist",
            },
        ],
    }

    prompts = build_scene_prompts(story)
    assert len(prompts) == 3

    # Verify character anchor is identically present in all 3 prompts
    anchor = build_character_anchor(story["character"])
    for p in prompts:
        assert anchor in p
        assert COMIC_VISUAL_STYLE in p
        assert "no text, no words" in p

    # Verify scene progression
    assert "cluttered desk" in prompts[0]
    assert "taking a deep breath" in prompts[1]
    assert "writing a revision checklist" in prompts[2]
