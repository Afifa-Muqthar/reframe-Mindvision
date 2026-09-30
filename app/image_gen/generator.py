"""
Image generation using stabilityai/sd-turbo, 256x256 by default, 1 step.

License note (say this in the viva if asked): SD-Turbo is under the
Stability AI Community/Research License - free to use for this academic
project, but non-commercial. It is also gated on Hugging Face: whoever
runs this needs a free HF account, to have clicked "accept" on the model
page, and to have run `huggingface-cli login` once before first download.

Runs in a background thread from the Flask route so it never blocks the
text response (see app/main.py /image_status/<session_id>).

Resolution note: the brief's spec is 512x512, but testing on the target
laptop (Intel i3, 3.77GB RAM total) showed the VAE-decode step segfaulting
at 512x512 -- almost certainly a failed native memory allocation under
memory pressure rather than a code bug (it crashed at the same point
regardless of thread count). Attention/VAE slicing and a default of
256x256 substantially cut peak memory; raise IMAGE_SIZE back to 512 only
on a machine with more headroom.
"""

import gc

import torch
from diffusers import AutoPipelineForText2Image

_MODEL_NAME = "stabilityai/sd-turbo"
IMAGE_SIZE = 256


class ImageGenerator:
    def __init__(self):
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            _MODEL_NAME, torch_dtype=torch.float32
        )
        self._pipe.to("cpu")
        # Cuts peak memory during the UNet and VAE-decode steps by
        # processing attention/VAE tiles sequentially instead of all at
        # once -- the difference between segfaulting and completing on a
        # low-RAM machine.
        self._pipe.enable_attention_slicing()
        self._pipe.enable_vae_slicing()

    def generate(self, prompt: str, output_path: str, num_inference_steps: int = 1,
                 size: int = IMAGE_SIZE, seed: int = None) -> str:
        # A fixed seed across the three storyboard panels (see
        # build_panel_prompts) keeps their color palette and style visually
        # cohesive even though SD-Turbo has no built-in way to keep the same
        # illustrated figure identical across separate generations.
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        image = self._pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # SD-Turbo is trained for guidance_scale=0
            height=size,
            width=size,
            generator=generator,
        ).images[0]
        image.save(output_path)
        return output_path

    def unload(self):
        del self._pipe
        gc.collect()
from app.image_gen.prompt_builder import (
    COMIC_VISUAL_STYLE,
    NEGATIVE_PROMPT,
    build_scene_prompts,
    build_character_anchor,
)


def build_image_prompt(case_frame, technique: dict) -> str:
    """
    Turns the case frame + chosen technique into a comic prompt.
    Kept for backward compatibility with scripts and tests.
    """
    return (
        f"A gentle comic illustration representing moving from {case_frame.core_emotion or 'worry'} "
        f"toward {technique['name'].lower()}, {COMIC_VISUAL_STYLE}"
    )


def build_panel_prompts(core_emotion: str, technique_name: str, story: dict = None) -> list:
    """
    Builds 3 prompts for the 3-panel comic strip (Problem -> Reframing -> Resolution).
    If a structured story is provided, delegates to build_scene_prompts;
    otherwise creates grounded fallback prompts with character consistency.
    """
    if story and isinstance(story, dict):
        return build_scene_prompts(story)

    emotion = core_emotion or "worried"
    character_anchor = "a young adult with short dark hair, wearing a dark hoodie and jeans, relatable student"
    return [
        f"{character_anchor}, sitting at a study desk overwhelmed by notes, hands on head, feeling {emotion}, {COMIC_VISUAL_STYLE}",
        f"{character_anchor}, sitting at desk pausing and taking a calm breath, looking at an open notebook, {COMIC_VISUAL_STYLE}",
        f"{character_anchor}, sitting upright with focused determination, writing a realistic plan, bright morning light, {COMIC_VISUAL_STYLE}",
    ]


def build_panel_subtitles(core_emotion: str, technique_name: str) -> list:
    """Short titles/captions for each panel's header."""
    return [
        f"1. PROBLEM (Feeling {core_emotion or 'this'})",
        f"2. REFRAME ({technique_name})",
        "3. RESOLUTION (One step forward)",
    ]
