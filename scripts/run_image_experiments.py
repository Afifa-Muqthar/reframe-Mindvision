"""
Controlled Empirical Image-Generation Experiments on SD-Turbo.

Investigates:
- Phase 3: Resolution experiment (256x256 vs 512x512)
- Phase 4: Inference steps experiment (1 vs 2 vs 4 steps at 512x512)
- Phase 5: Scene differentiation experiment (5 distinct concrete actions)
- Phase 7: Prompt length experiment (concise vs current vs long)
- Phase 8: Negative prompt experiment (does negative_prompt affect SD-Turbo?)
"""

import os
import sys
import time
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ssl_workaround import configure_hf_ssl_verify
configure_hf_ssl_verify()

from diffusers import AutoPipelineForText2Image

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experiments")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "stabilityai/sd-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

print(f"Loading pipeline on {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
pipe.to(DEVICE)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

results = {}

def measure_generation(name, prompt, steps=1, size=512, seed=42, guidance_scale=0.0, negative_prompt=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    
    t0 = time.perf_counter()
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "height": size,
        "width": size,
        "generator": gen,
    }
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    image = pipe(**kwargs).images[0]
    t1 = time.perf_counter()

    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    elapsed_s = t1 - t0

    out_file = os.path.join(OUT_DIR, f"{name}.png")
    image.save(out_file)

    res = {
        "name": name,
        "file": out_file,
        "steps": steps,
        "size": f"{size}x{size}",
        "time_seconds": round(elapsed_s, 3),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    print(f"[{name}] {size}x{size}, {steps} steps -> {res['time_seconds']}s, {res['peak_vram_mb']} MB VRAM")
    return res

# -------------------------------------------------------------
# PHASE 3: RESOLUTION EXPERIMENT (256x256 vs 512x512)
# -------------------------------------------------------------
print("\n--- PHASE 3: RESOLUTION EXPERIMENT ---")
base_prompt = "a young woman with short dark hair, sitting on edge of bed holding smartphone with both hands, quiet bedroom, comic illustration"
results["phase3_256"] = measure_generation("exp_phase3_256", base_prompt, steps=2, size=256, seed=100)
results["phase3_512"] = measure_generation("exp_phase3_512", base_prompt, steps=2, size=512, seed=100)

# -------------------------------------------------------------
# PHASE 4: INFERENCE STEPS (1 vs 2 vs 4 steps at 512x512)
# -------------------------------------------------------------
print("\n--- PHASE 4: INFERENCE STEPS EXPERIMENT (512x512) ---")
results["phase4_step1"] = measure_generation("exp_phase4_step1", base_prompt, steps=1, size=512, seed=100)
results["phase4_step2"] = measure_generation("exp_phase4_step2", base_prompt, steps=2, size=512, seed=100)
results["phase4_step4"] = measure_generation("exp_phase4_step4", base_prompt, steps=4, size=512, seed=100)

# -------------------------------------------------------------
# PHASE 5: SCENE DIFFERENTIATION (5 distinct concrete actions)
# -------------------------------------------------------------
print("\n--- PHASE 5: SCENE DIFFERENTIATION EXPERIMENT ---")
actions = [
    ("young woman sitting on edge of bed holding smartphone with both hands, quiet bedroom", "action_A_sitting_phone"),
    ("young woman standing beside a desk writing in a notebook, quiet bedroom", "action_B_standing_writing"),
    ("young woman walking toward an open bedroom doorway, quiet bedroom", "action_C_walking_doorway"),
    ("young woman sitting beside another person and talking, quiet room", "action_D_talking_person"),
    ("young woman standing outside in daylight looking toward the street", "action_E_outside_street"),
]
style_tag = ", vibrant modern comic book illustration, crisp ink line art, colorful flat cel shading"
for prompt_act, label in actions:
    full_p = f"{prompt_act}{style_tag}"
    # Use fixed seed=200 to test whether differing prompts overcome seed anchoring
    results[f"phase5_{label}_fixed_seed"] = measure_generation(f"exp_phase5_{label}_seed200", full_p, steps=2, size=512, seed=200)

# Also test action_B with varied seed=201
results["phase5_action_B_varied_seed"] = measure_generation("exp_phase5_action_B_seed201", f"{actions[1][0]}{style_tag}", steps=2, size=512, seed=201)

# -------------------------------------------------------------
# PHASE 7: PROMPT LENGTH EXPERIMENT
# -------------------------------------------------------------
print("\n--- PHASE 7: PROMPT LENGTH EXPERIMENT ---")
prompt_concise = "young woman with short dark hair standing by open window looking at sky, comic book art"
prompt_current = (
    "young adult with short dark hair, wearing a dark gray sweater and dark pants, medium shot, eye level, "
    "standing upright by the window taking a slow deep breath, peaceful posture, crisp bright natural daylight, "
    "Hopeful, purposeful, grounded, vibrant modern comic book illustration, crisp ink line art, bold outlines, "
    "expressive cartoon character, colorful flat cel shading, graphic novel panel art, clean composition, no text, no words, no watermark"
)
prompt_extreme_long = (
    "young adult with short dark hair, wearing a dark gray sweater and dark pants, relatable student finding clarity, "
    "medium shot, eye level angle, composition centered beside open wooden window frame, character standing tall with upright balanced spine, "
    "open relaxed shoulders, hands resting peacefully on window sill, pensive gentle smile, looking out at blue sky and morning sun, "
    "bedroom interior with wooden study desk, warm mug of tea, soft morning sunlight casting long shadows across floorboards, "
    "feeling relieved and unburdened by past doubts, vibrant modern comic book illustration, crisp ink line art, bold outlines, "
    "expressive cartoon character, colorful flat cel shading, graphic novel panel art, clean composition, highly detailed masterpiece, "
    "no text, no words, no watermark, no signatures, perfect anatomy"
)

results["phase7_concise"] = measure_generation("exp_phase7_concise", prompt_concise, steps=2, size=512, seed=300)
results["phase7_current"] = measure_generation("exp_phase7_current", prompt_current, steps=2, size=512, seed=300)
results["phase7_long"] = measure_generation("exp_phase7_long", prompt_extreme_long, steps=2, size=512, seed=300)

# -------------------------------------------------------------
# PHASE 8: NEGATIVE PROMPT EXPERIMENT
# -------------------------------------------------------------
print("\n--- PHASE 8: NEGATIVE PROMPT EXPERIMENT ---")
neg_prompt = "photorealistic, real photograph, 3d render, deformed, ugly, extra limbs, bad anatomy"
results["phase8_no_neg"] = measure_generation("exp_phase8_no_neg", base_prompt, steps=2, size=512, seed=400, guidance_scale=0.0, negative_prompt=None)
results["phase8_with_neg_scale0"] = measure_generation("exp_phase8_with_neg_scale0", base_prompt, steps=2, size=512, seed=400, guidance_scale=0.0, negative_prompt=neg_prompt)
# Also test if guidance_scale=1.0 or 1.5 causes artifacts in SD-Turbo
results["phase8_with_neg_scale1_5"] = measure_generation("exp_phase8_with_neg_scale1_5", base_prompt, steps=2, size=512, seed=400, guidance_scale=1.5, negative_prompt=neg_prompt)

with open(os.path.join(OUT_DIR, "experiment_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
