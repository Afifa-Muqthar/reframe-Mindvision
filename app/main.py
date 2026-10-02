"""
Flask orchestrator for the AI Mental Reframing Interface.

Workflow:
  User input (text/audio) -> (ASR if audio) -> Safety Gate ->
  Cognitive Distortion Detection -> Emotion Classification ->
  Context Engine (Case Frame + Recurring Theme) -> Principle Selector ->
  Structured 3-Stage Narrative (Problem -> Reframing -> Resolution) ->
  Database Save -> Connected TTS Voice Narration ->
  Process-Isolated Comic Strip Generation (SD-Turbo + Pillow Comic Compositor) ->
  Asynchronous Frontend Polling.

RAM discipline:
  spaCy and the emotion classifier stay resident across requests.
  Whisper, FLAN-T5, and Kokoro are loaded only when needed and released immediately.
  SD-Turbo runs isolated in a separate worker subprocess.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import subprocess
import sys
import threading
import uuid

from dotenv import load_dotenv
load_dotenv()

from app.ssl_workaround import configure_hf_ssl_verify
configure_hf_ssl_verify()

from flask import Flask, request, jsonify, render_template, send_from_directory

from app.safety.safety_gate import screen_for_crisis, get_helpline_message
from app.nlp.distortions import detect_distortions
from app.emotion.classifier import EmotionClassifier
from app.context_engine.case_frame import CaseFrame
from app.psychology import parse_situation, select_strategy
from app.principle_selector.selector import select_techniques
from app.narrative_gen.generator import NarrativeGenerator, format_narrative_text, parse_narrative_parts
from app.image_gen.prompt_builder import build_scene_prompts
from app.db.schema import init_db
from app.db import repository

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, "data", "sessions")
os.makedirs(MEDIA_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
init_db()

# Emotion classifier + spaCy stay resident
_emotion_classifier = EmotionClassifier()

# Tracks in-flight background image generation per session id
_image_status = {}  # session_id(str) -> {"status": "pending"|"done"|"error", "path": str|None}
_image_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/journey")
def journey():
    sessions = repository.list_sessions()
    recurring = repository.find_recurring_themes()
    return render_template("journey.html", sessions=sessions, recurring=recurring)


@app.route("/media/<path:filename>")
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)


@app.route("/process", methods=["POST"])
def process():
    text = request.form.get("text", "").strip()
    audio_file = request.files.get("audio")

    if audio_file and not text:
        text = _transcribe_audio(audio_file)

    if not text:
        return jsonify({"error": "No text or audio provided."}), 400

    # 1. Safety gate — ALWAYS FIRST, fail-safe.
    if screen_for_crisis(text):
        return jsonify({
            "crisis": True,
            "message": get_helpline_message(),
        })

    # 2. Psychological Reasoning Layer: Situation & Pattern Parsing
    psych_profile = parse_situation(text)

    # 3. Emotion detection (resident ONNX classifier)
    emotion_scores, core_emotion = _emotion_classifier.classify(text)
    if psych_profile.is_external_threat and core_emotion in ("neutral", "joy"):
        core_emotion = "fear"

    # 4. Cognitive distortion scan (only for internal thoughts, NOT external harm)
    distortions = []
    if not psych_profile.is_external_threat:
        distortions = detect_distortions(text)

    # 5. Evidence-informed strategy selection (APA, SAMHSA, NIMH)
    strategy = select_strategy(psych_profile)

    # 6. Context Engine: assemble Case Frame with psychological reasoning
    reasoning_dict = psych_profile.structured_reasoning()
    reasoning_dict.update({
        "what_may_be_happening": f"{psych_profile.trigger} → {psych_profile.pattern}",
        "what_you_can_control": strategy.what_is_controllable,
        "what_is_not_controllable": strategy.what_is_not_controllable,
        "reframe": strategy.reframe,
        "next_step": strategy.concrete_action,
    })

    case_frame = CaseFrame(
        raw_text=text,
        distortions=distortions,
        emotion_scores=emotion_scores,
        core_emotion=core_emotion,
        situation_type=psych_profile.situation_id,
        trigger=psych_profile.trigger,
        pattern=psych_profile.pattern,
        is_external_threat=psych_profile.is_external_threat,
        controllable=strategy.what_is_controllable,
        uncontrollable=strategy.what_is_not_controllable,
        strategy=strategy.to_dict(),
        specific_reframe=strategy.reframe,
        concrete_action=strategy.concrete_action,
        reasoning=reasoning_dict,
    )
    case_frame.build_summary()
    case_frame.recurring_theme_note = repository.build_recurring_theme_note()

    primary_technique = {
        "name": strategy.name,
        "citation": strategy.citation,
        "description": strategy.mechanism,
    }

    # 7. Grounded Narrative & Observable Visual Staging (Stage 1-4)
    story = None
    use_grounded = True
    try:
        from app.psychology.case_representation import extract_case_representation
        from app.psychology.case_strategy_selector import CaseStrategySelector
        from app.narrative_gen.grounded_generator import GroundedNarrativeGenerator
        from app.narrative_gen.scene_planner import ScenePlanner

        case_rep = extract_case_representation(text)
        strategy_selector = CaseStrategySelector()
        selected_strategy = strategy_selector.select(case_rep)

        grounded_gen = GroundedNarrativeGenerator()
        raw_story = grounded_gen.generate(case_rep, selected_strategy)

        # Stage 4: Enrich scenes with grounded physical visual staging
        planner = ScenePlanner()
        enriched_scenes = planner.plan_grounded_scenes(raw_story, case_rep, selected_strategy)
        raw_story["scenes"] = enriched_scenes
        story = raw_story

        # Align case_frame and technique with grounded psychological reasoning
        controllable_str = (
            "; ".join(case_rep.controllability.potentially_controllable)
            if case_rep.controllability.potentially_controllable
            else "Focus on personal boundaries, self-advocacy, and emotional safety."
        )
        uncontrollable_str = (
            "; ".join(case_rep.controllability.potentially_uncontrollable)
            if case_rep.controllability.potentially_uncontrollable
            else "Past events and external actions of others."
        )
        # Ensure reasoning object contains only the 5 intended grounded fields
        case_frame.reasoning = {
            "what_may_be_happening": selected_strategy.rationale,
            "what_you_can_control": controllable_str,
            "what_is_not_controllable": uncontrollable_str,
            "reframe": selected_strategy.core_message,
            "next_step": selected_strategy.suggested_step,
        }
        case_frame.controllable = controllable_str
        case_frame.uncontrollable = uncontrollable_str
        case_frame.specific_reframe = selected_strategy.core_message
        case_frame.concrete_action = selected_strategy.suggested_step
        case_frame.summary = case_rep.descriptive_summary
        case_frame.pattern = f"Clinical Strategy: {selected_strategy.modality.replace('_', ' ').title()}"
        case_frame.is_external_threat = (
            any(a.dimension in ("systemic_injustice", "relational_harm", "medical_adversity", "external_threat") for a in case_rep.inferred_appraisals)
            or selected_strategy.strategy_id in ("injustice_validation_options", "bodily_limitation_compassion")
        )
        primary_technique = {
            "name": selected_strategy.modality.replace("_", " ").title(),
            "citation": selected_strategy.clinical_framework,
            "description": selected_strategy.rationale,
        }
    except Exception as exc:
        import traceback
        sys.stderr.write(f"[PIPELINE FALLBACK] Grounded pipeline failed, using legacy fallback: {exc}\n")
        traceback.print_exc(file=sys.stderr)
        use_grounded = False

    if not use_grounded or story is None:
        generator = NarrativeGenerator()
        try:
            story = generator.generate(case_frame, primary_technique)
        finally:
            generator.unload()

    formatted_narrative = format_narrative_text(story)

    # 8. Save to DB now (audio/image paths updated asynchronously).
    session_id = repository.save_session(case_frame, formatted_narrative)

    # 9. Connected TTS voice narration (tells the cohesive psychological story).
    voice_script = (story.get("voice_script") if story else None) or strategy.voice_script or formatted_narrative
    audio_path = _run_tts(session_id, voice_script)
    if audio_path:
        repository.update_audio_path(session_id, audio_path)

    # 10. Launch process-isolated comic image generator with structured config.
    _image_status[str(session_id)] = {"status": "pending", "path": None}
    scene_prompts = build_scene_prompts(story)
    output_image_path = os.path.join(MEDIA_DIR, f"session_{session_id}.png")

    comic_title = f"REFRAME COMIC: {story.get('title', story.get('context', 'MENTAL REFRAMING')).upper()}"
    job_config = {
        "session_id": session_id,
        "prompts": scene_prompts,
        "scenes": story.get("scenes", []),
        "output_path": output_image_path,
        "seed": session_id,
        "num_inference_steps": 2,
        "title": comic_title,
    }
    config_path = os.path.join(MEDIA_DIR, f"job_{session_id}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(job_config, f, indent=2)

    threading.Thread(
        target=_generate_image_background,
        args=(session_id, config_path, output_image_path),
        daemon=True,
    ).start()

    response = {
        "crisis": False,
        "session_id": session_id,
        "summary": case_frame.summary,
        "distortions": case_frame.distortions,
        "core_emotion": case_frame.core_emotion,
        "pattern": case_frame.pattern,
        "is_external_threat": case_frame.is_external_threat,
        "reasoning": case_frame.reasoning,
        "technique": {"name": primary_technique["name"], "citation": primary_technique["citation"]},
        "narrative": formatted_narrative,
        "story": story,
        "audio_url": f"/media/{os.path.basename(audio_path)}" if audio_path else None,
        "image_status_url": f"/image_status/{session_id}",
        "pipeline": "grounded_pipeline_v1" if use_grounded else "legacy_pipeline",
    }
    return jsonify(response)


@app.route("/image_status/<int:session_id>")
def image_status(session_id):
    status = _image_status.get(str(session_id), {"status": "unknown", "path": None})
    resp = {"status": status["status"]}
    if status["status"] == "done" and status["path"]:
        resp["image_url"] = f"/media/{os.path.basename(status['path'])}"
    elif status["status"] == "error":
        resp["error"] = status.get("error", "Image generation failed.")
    return jsonify(resp)


# ---- internal helpers ----

def _transcribe_audio(audio_file) -> str:
    from app.asr.transcriber import Transcriber

    tmp_path = os.path.join(MEDIA_DIR, f"upload_{uuid.uuid4().hex}.wav")
    audio_file.save(tmp_path)
    transcriber = Transcriber()
    try:
        text = transcriber.transcribe(tmp_path)
    finally:
        transcriber.unload()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return text


def _run_tts(session_id: int, script_text: str) -> str:
    from app.tts.narrator import Narrator

    output_path = os.path.join(MEDIA_DIR, f"session_{session_id}.wav")
    narrator = Narrator()
    try:
        narrator.narrate(script_text, output_path)
    except Exception:
        return None
    finally:
        narrator.unload()
    return output_path


def _generate_image_background(session_id: int, config_path: str, output_path: str):
    """
    Spawns the dedicated image worker in an isolated child process using
    structured JSON arguments. Protects the parent Flask process from VAE-decode
    native memory segmentation faults. Uses a thread lock to serialize worker
    executions and prevent GPU memory contention.
    """
    with _image_lock:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "app.image_gen.worker", config_path],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0 and os.path.exists(output_path):
                repository.update_image_path(session_id, output_path)
                _image_status[str(session_id)] = {"status": "done", "path": output_path}
            else:
                sys.stderr.write(
                    f"[IMAGE WORKER ERROR] Session {session_id} code={result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}\n"
                )
                err_output = result.stderr[-500:] if result.stderr else "Image generation worker failed."
                _image_status[str(session_id)] = {"status": "error", "path": None, "error": err_output}
        except Exception as e:
            _image_status[str(session_id)] = {"status": "error", "path": None, "error": str(e)}
        finally:
            if os.path.exists(config_path):
                try:
                    os.remove(config_path)
                except OSError:
                    pass


if __name__ == "__main__":
    app.run(debug=bool(os.environ.get("FLASK_DEBUG")), port=5000)
