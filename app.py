import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Callable
from uuid import uuid4

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from imageio_ffmpeg import get_ffmpeg_exe
from openai import OpenAI
from werkzeug.utils import secure_filename

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
    ".ogg",
}
VALID_TASK_MODES = {"transcribe", "translate"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", token_hex(32))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("WHISPER_MODEL", "whisper-1")
NOTES_MODEL = os.getenv("NOTES_MODEL", "gpt-4o-mini")

JOB_WORKERS = int(os.getenv("JOB_WORKERS", "2"))
OPENAI_MAX_BYTES = int(os.getenv("OPENAI_MAX_BYTES", str(25 * 1024 * 1024)))
CHUNK_MINUTES = int(os.getenv("CHUNK_MINUTES", "10"))
NOTES_CHUNK_CHARS = int(os.getenv("NOTES_CHUNK_CHARS", "12000"))
CHUNK_TRANSCRIBE_WORKERS = int(os.getenv("CHUNK_TRANSCRIBE_WORKERS", "3"))
ENABLE_DIARIZATION_BETA = os.getenv("ENABLE_DIARIZATION_BETA", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUPPORTED_DIRECT_EXTS = {".m4a", ".mp3", ".wav"}
STAGE_LABELS = {
    "preprocess": "Preprocess audio",
    "transcribe": "Transcribe",
    "diarize": "Speaker attribution",
    "notes": "Generate notes",
    "finalize": "Finalize outputs",
}

executor = ThreadPoolExecutor(max_workers=JOB_WORKERS)
jobs_lock = threading.Lock()
jobs: dict[str, dict[str, object]] = {}

try:
    FFMPEG_EXE = get_ffmpeg_exe()
except Exception:  # noqa: BLE001
    FFMPEG_EXE = None

FFPROBE_EXE = None
if FFMPEG_EXE:
    ffmpeg_path = Path(FFMPEG_EXE)
    ffprobe_candidate = ffmpeg_path.with_name(ffmpeg_path.name.replace("ffmpeg", "ffprobe"))
    if ffprobe_candidate.exists():
        FFPROBE_EXE = str(ffprobe_candidate)


def _is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _normalize_task_mode(task_mode: str | None) -> str:
    if task_mode in VALID_TASK_MODES:
        return task_mode
    return "transcribe"


def _new_job(
    *,
    original_filename: str,
    audio_path: str,
    task_mode: str,
    language: str | None,
    with_timestamps: bool,
    enable_speaker: bool,
    generate_notes: bool,
    notes_in_english: bool,
) -> str:
    job_id = uuid4().hex
    created_at = datetime.now(timezone.utc)
    with jobs_lock:
        stage_enabled = {
            "preprocess": True,
            "transcribe": True,
            "diarize": enable_speaker,
            "notes": generate_notes,
            "finalize": True,
        }
        stage_order = ["preprocess", "transcribe", "diarize", "notes", "finalize"]
        stage_progress = {
            stage: (0 if enabled else 100) for stage, enabled in stage_enabled.items()
        }
        stage_status = {
            stage: ("pending" if enabled else "skipped")
            for stage, enabled in stage_enabled.items()
        }
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "status_detail": "Waiting in queue",
            "overall_progress": 0,
            "stage_enabled": stage_enabled,
            "stage_progress": stage_progress,
            "stage_status": stage_status,
            "stage_labels": STAGE_LABELS,
            "stage_order": stage_order,
            "created_at": created_at,
            "original_filename": original_filename,
            "audio_path": audio_path,
            "task_mode": task_mode,
            "language": language,
            "with_timestamps": with_timestamps,
            "enable_speaker": enable_speaker,
            "generate_notes": generate_notes,
            "notes_in_english": notes_in_english,
            "transcript_text": None,
            "transcript_segments": [],
            "transcript_txt": None,
            "transcript_json": None,
            "notes_txt": None,
            "notes_json": None,
            "warnings": [],
            "error": None,
        }
    return job_id


def _get_job(job_id: str) -> dict[str, object] | None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return None
        return dict(job)


def _set_job_fields(job_id: str, **kwargs: object) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def _recompute_overall_progress(job_id: str) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        job = jobs[job_id]
        stage_enabled = job.get("stage_enabled", {})
        stage_progress = job.get("stage_progress", {})
        if not isinstance(stage_enabled, dict) or not isinstance(stage_progress, dict):
            job["overall_progress"] = 0
            return

        enabled_stages = [k for k, v in stage_enabled.items() if bool(v)]
        if not enabled_stages:
            job["overall_progress"] = 0
            return

        total = 0.0
        for stage in enabled_stages:
            value = stage_progress.get(stage, 0)
            try:
                total += max(0, min(100, int(value)))
            except Exception:  # noqa: BLE001
                total += 0
        job["overall_progress"] = int(round(total / len(enabled_stages)))


def _set_stage_state(
    job_id: str,
    stage: str,
    *,
    progress: int | None = None,
    status: str | None = None,
    detail: str | None = None,
) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        job = jobs[job_id]
        stage_progress = job.get("stage_progress")
        stage_status = job.get("stage_status")
        if not isinstance(stage_progress, dict):
            stage_progress = {}
        if not isinstance(stage_status, dict):
            stage_status = {}

        if progress is not None:
            stage_progress[stage] = max(0, min(100, int(progress)))
            job["stage_progress"] = stage_progress
        if status is not None:
            stage_status[stage] = status
            job["stage_status"] = stage_status
        if detail is not None:
            job["status_detail"] = detail

    _recompute_overall_progress(job_id)


def _append_job_warning(job_id: str, message: str) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        warnings = jobs[job_id].get("warnings")
        if not isinstance(warnings, list):
            warnings = []
        warnings.append(message)
        jobs[job_id]["warnings"] = warnings


def _run_whisper_request(
    audio_path: str,
    task_mode: str,
    language: str | None,
    include_segments: bool,
) -> tuple[str, list[dict[str, object]]]:
    options: dict[str, object] = {"model": MODEL_NAME}

    # In transcribe mode, language is optional and only used as a hint.
    if task_mode == "transcribe" and language:
        options["language"] = language

    options["response_format"] = "verbose_json" if include_segments else "text"

    with open(audio_path, "rb") as audio_file:
        if task_mode == "translate":
            result = client.audio.translations.create(file=audio_file, **options)
        else:
            result = client.audio.transcriptions.create(file=audio_file, **options)

    if isinstance(result, str):
        return result.strip(), []

    text = str(getattr(result, "text", "")).strip()
    if not include_segments:
        return text, []

    raw_segments = getattr(result, "segments", None) or []
    segments: list[dict[str, object]] = []
    for segment in raw_segments:
        seg_text = str(getattr(segment, "text", "")).strip()
        if not seg_text:
            continue
        segments.append(
            {
                "start": float(getattr(segment, "start", 0.0)),
                "end": float(getattr(segment, "end", 0.0)),
                "text": seg_text,
            }
        )

    if not text and segments:
        text = " ".join(str(seg["text"]) for seg in segments).strip()
    return text, segments


def _parse_ffmpeg_time_to_seconds(text: str) -> float | None:
    match = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def _run_ffmpeg_with_progress(
    command: list[str],
    expected_seconds: float | None,
    progress_callback: Callable[[int], None] | None = None,
) -> None:
    if progress_callback:
        progress_callback(1)
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if process.stderr:
        for line in process.stderr:
            if not progress_callback or not expected_seconds or expected_seconds <= 0:
                continue
            t = _parse_ffmpeg_time_to_seconds(line)
            if t is None:
                continue
            pct = int(min(95, max(1, (t / expected_seconds) * 100)))
            progress_callback(pct)

    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"FFmpeg command failed with exit code {exit_code}")
    if progress_callback:
        progress_callback(100)


def _probe_audio_metadata(audio_path: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if not FFPROBE_EXE:
        return metadata

    command = [
        FFPROBE_EXE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        audio_path,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        parsed = json.loads(result.stdout or "{}")
    except Exception:  # noqa: BLE001
        return metadata

    streams = parsed.get("streams", []) if isinstance(parsed, dict) else []
    if isinstance(streams, list):
        audio_stream = next(
            (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"),
            {},
        )
    else:
        audio_stream = {}
    format_block = parsed.get("format", {}) if isinstance(parsed, dict) else {}
    if not isinstance(format_block, dict):
        format_block = {}

    def _to_int(value: object) -> int | None:
        try:
            return int(str(value))
        except Exception:  # noqa: BLE001
            return None

    def _to_float(value: object) -> float | None:
        try:
            return float(str(value))
        except Exception:  # noqa: BLE001
            return None

    metadata["codec_name"] = audio_stream.get("codec_name")
    metadata["sample_rate"] = _to_int(audio_stream.get("sample_rate"))
    metadata["channels"] = _to_int(audio_stream.get("channels"))
    metadata["format_name"] = format_block.get("format_name")
    metadata["duration"] = _to_float(format_block.get("duration"))
    return metadata


def _should_preprocess(
    audio_path: str,
    *,
    enable_speaker: bool,
    metadata: dict[str, object],
) -> tuple[bool, str]:
    if enable_speaker:
        return True, "speaker_diarization_enabled"

    ext = Path(audio_path).suffix.lower()
    sample_rate = metadata.get("sample_rate")
    channels = metadata.get("channels")

    if ext not in SUPPORTED_DIRECT_EXTS:
        return True, f"unsupported_format:{ext or 'unknown'}"
    if not isinstance(sample_rate, int) or sample_rate < 16000 or sample_rate > 48000:
        return True, f"unsupported_sample_rate:{sample_rate}"
    if not isinstance(channels, int) or channels <= 0 or channels > 2:
        return True, f"unsupported_channels:{channels}"
    return False, "already_supported"


def _preprocess_audio(
    audio_path: str,
    expected_seconds: float | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[str, str | None]:
    if not FFMPEG_EXE:
        return audio_path, "FFmpeg unavailable, skipping audio preprocessing."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        processed_path = tmp.name

    command = [
        FFMPEG_EXE,
        "-hide_banner",
        "-y",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        processed_path,
    ]
    try:
        _run_ffmpeg_with_progress(command, expected_seconds=expected_seconds, progress_callback=progress_callback)
        return processed_path, None
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(processed_path):
            os.remove(processed_path)
        return audio_path, f"Audio preprocessing failed, using original upload. ({exc})"


def _segment_audio_chunks(
    source_audio: str,
    output_dir: str,
    *,
    needs_conversion: bool,
) -> list[tuple[int, str, float]]:
    if not FFMPEG_EXE:
        raise RuntimeError("FFmpeg is not available for chunking.")

    chunk_seconds = max(CHUNK_MINUTES, 1) * 60
    ext = ".wav" if needs_conversion else (Path(source_audio).suffix.lower() or ".m4a")
    out_pattern = str(Path(output_dir) / f"chunk_%05d{ext}")

    command = [
        FFMPEG_EXE,
        "-hide_banner",
        "-y",
        "-i",
        source_audio,
        "-vn",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
    ]
    if needs_conversion:
        command.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
    else:
        command.extend(["-c:a", "copy"])
    command.append(out_pattern)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        if not needs_conversion:
            # Copy segmentation can fail on some containers; retry with PCM conversion.
            return _segment_audio_chunks(source_audio, output_dir, needs_conversion=True)
        raise

    chunk_files = sorted(Path(output_dir).glob("chunk_*"))
    if not chunk_files:
        raise RuntimeError("No chunks were generated from audio.")

    chunk_list: list[tuple[int, str, float]] = []
    for index, chunk_file in enumerate(chunk_files):
        chunk_list.append((index, str(chunk_file), float(index * chunk_seconds)))
    return chunk_list


def _transcribe_with_chunking(
    audio_path: str,
    task_mode: str,
    language: str | None,
    include_segments: bool,
    duration_seconds: float | None,
    force_conversion_chunks: bool,
    progress_callback: Callable[[str, int], None] | None = None,
) -> tuple[str, list[dict[str, object]]]:
    original_size = os.path.getsize(audio_path)
    if original_size <= OPENAI_MAX_BYTES and not force_conversion_chunks:
        if progress_callback:
            progress_callback("Transcribing single chunk", 20)
        started = time.perf_counter()
        text, segments = _run_whisper_request(
            audio_path=audio_path,
            task_mode=task_mode,
            language=language,
            include_segments=include_segments,
        )
        logger.info("transcribe single_chunk elapsed=%.2fs", time.perf_counter() - started)
        if progress_callback:
            progress_callback("Transcribing single chunk", 100)
        return text, segments

    with tempfile.TemporaryDirectory() as chunk_dir:
        segment_started = time.perf_counter()
        chunk_list = _segment_audio_chunks(
            audio_path,
            chunk_dir,
            needs_conversion=force_conversion_chunks,
        )
        logger.info(
            "chunk_segmentation total_chunks=%s elapsed=%.2fs",
            len(chunk_list),
            time.perf_counter() - segment_started,
        )

        total_chunks = max(1, len(chunk_list))
        logger.info(
            "chunk_pipeline total_chunks=%s workers=%s force_conversion=%s",
            total_chunks,
            CHUNK_TRANSCRIBE_WORKERS,
            force_conversion_chunks,
        )

        results: dict[int, tuple[float, str, list[dict[str, object]]]] = {}
        errors: list[str] = []
        completed = 0

        def submit_chunk(
            executor: ThreadPoolExecutor,
            idx: int,
            chunk_path: str,
            offset_seconds: float,
        ):
            def worker() -> tuple[int, float, str, list[dict[str, object]]]:
                transcribe_start = time.perf_counter()
                chunk_size = os.path.getsize(chunk_path)
                if chunk_size > OPENAI_MAX_BYTES:
                    raise ValueError(
                        f"Chunk {idx + 1} exceeds OpenAI API limit ({chunk_size} bytes). "
                        "Reduce CHUNK_MINUTES."
                    )
                text_part, segment_part = _run_whisper_request(
                    audio_path=chunk_path,
                    task_mode=task_mode,
                    language=language,
                    include_segments=include_segments,
                )
                transcribe_elapsed = time.perf_counter() - transcribe_start
                logger.info(
                    "chunk index=%s transcribe=%.2fs size=%s",
                    idx,
                    transcribe_elapsed,
                    chunk_size,
                )
                return idx, offset_seconds, text_part, segment_part

            return executor.submit(worker)

        with ThreadPoolExecutor(max_workers=max(1, CHUNK_TRANSCRIBE_WORKERS)) as pool:
            futures: dict[object, tuple[int, str, float]] = {}
            for idx, chunk_path, offset_seconds in chunk_list:
                fut = submit_chunk(pool, idx, chunk_path, offset_seconds)
                futures[fut] = (idx, chunk_path, offset_seconds)
                if progress_callback:
                    queued = len(futures)
                    prep_pct = int((queued / total_chunks) * 35)
                    progress_callback(
                        f"Preparing/transcribing chunks ({queued}/{total_chunks} queued)",
                        prep_pct,
                    )

            for fut in as_completed(futures):
                idx, chunk_path, _offset = futures[fut]
                try:
                    r_idx, offset_seconds, text_part, segment_part = fut.result()
                    results[r_idx] = (offset_seconds, text_part, segment_part)
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
                finally:
                    completed += 1
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    if progress_callback:
                        pct = 35 + int((completed / total_chunks) * 65)
                        progress_callback(
                            f"Transcribing chunks ({completed}/{total_chunks} completed)",
                            min(100, pct),
                        )

    if errors:
        raise RuntimeError(errors[0])

    parts: list[str] = []
    all_segments: list[dict[str, object]] = []
    for idx in sorted(results.keys()):
        offset_seconds, text_part, segment_part = results[idx]
        if text_part.strip():
            parts.append(text_part.strip())
        for segment in segment_part:
            all_segments.append(
                {
                    "start": float(segment["start"]) + offset_seconds,
                    "end": float(segment["end"]) + offset_seconds,
                    "text": segment["text"],
                }
            )

    full_text = "\n\n".join(parts).strip()
    return full_text, all_segments


def _run_diarization(audio_path: str) -> list[dict[str, object]]:
    if not ENABLE_DIARIZATION_BETA:
        raise RuntimeError("Speaker attribution is disabled. Set ENABLE_DIARIZATION_BETA=true to enable.")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for pyannote diarization.")

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pyannote.audio is not installed. Install optional diarization dependencies."
        ) from exc

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline(audio_path)

    raw_turns: list[dict[str, object]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    # Map pyannote labels to SPEAKER_XX for stable output.
    mapping: dict[str, str] = {}
    mapped_turns: list[dict[str, object]] = []
    for row in raw_turns:
        original = str(row["speaker"])
        if original not in mapping:
            mapping[original] = f"SPEAKER_{len(mapping):02d}"
        mapped_turns.append(
            {
                "start": row["start"],
                "end": row["end"],
                "speaker": mapping[original],
            }
        )
    return mapped_turns


def _annotate_segments_with_speaker(
    segments: list[dict[str, object]],
    speaker_turns: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not segments or not speaker_turns:
        return segments

    output: list[dict[str, object]] = []
    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        best_speaker = None
        best_overlap = 0.0

        for turn in speaker_turns:
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
            overlap = max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        updated = dict(segment)
        if best_speaker and best_overlap > 0:
            updated["speaker"] = str(best_speaker)
        output.append(updated)
    return output


def _build_transcript_txt(
    transcript_text: str,
    segments: list[dict[str, object]],
    with_timestamps: bool,
) -> str:
    if not segments:
        return transcript_text.strip()

    lines: list[str] = []
    for segment in segments:
        seg_text = str(segment.get("text", "")).strip()
        if not seg_text:
            continue
        prefix_parts: list[str] = []
        if with_timestamps:
            prefix_parts.append(
                f"[{_format_timestamp(float(segment['start']))} - {_format_timestamp(float(segment['end']))}]"
            )
        if segment.get("speaker"):
            prefix_parts.append(f"{segment['speaker']}:")

        if prefix_parts:
            lines.append(f"{' '.join(prefix_parts)} {seg_text}")
        else:
            lines.append(seg_text)
    return "\n".join(lines).strip()


def _split_text_for_notes(text: str, max_chars: int) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + 1
        if current and current_len + para_len > max_chars:
            chunks.append("\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def _parse_json_output(content: str) -> dict[str, object]:
    content = content.strip()
    if not content:
        return {}
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:  # noqa: BLE001
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(content[start : end + 1])
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception:  # noqa: BLE001
                return {"raw": content}
    return {"raw": content}


def _chat_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
    response = client.chat.completions.create(
        model=NOTES_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return _parse_json_output(content)


def _generate_chunk_summary(chunk_text: str, notes_in_english: bool) -> dict[str, object]:
    language_instruction = (
        "Write output in English. If a decision/action appears in a non-English utterance, "
        "include short evidence with original snippet and English paraphrase where possible."
        if notes_in_english
        else "Write output in the dominant language of the chunk."
    )
    system_prompt = (
        "You summarize meeting transcript chunks. Return strict JSON only with keys: "
        "key_points (array), decisions (array), actions (array of objects with keys owner, action, deadline, evidence_timestamp), "
        "open_questions (array), notable_quotes (array). Use empty arrays when unknown."
    )
    user_prompt = f"{language_instruction}\n\nTranscript chunk:\n{chunk_text}"
    return _chat_json(system_prompt, user_prompt)


def _generate_final_notes(
    chunk_summaries: list[dict[str, object]],
    notes_in_english: bool,
) -> dict[str, object]:
    language_instruction = (
        "Write output in English."
        if notes_in_english
        else "Write output in the same language as the majority of the meeting."
    )
    system_prompt = (
        "You synthesize meeting chunk summaries. Return strict JSON only with keys: "
        "executive_summary (5-10 bullets), decisions (array), action_items (array), "
        "risks_blockers (array), follow_ups_next_meeting (array)."
    )
    user_prompt = (
        f"{language_instruction}\n\nChunk summaries JSON:\n"
        f"{json.dumps(chunk_summaries, ensure_ascii=False)}"
    )
    result = _chat_json(system_prompt, user_prompt)
    result["chunk_summaries"] = chunk_summaries
    return result


def _notes_json_to_text(notes: dict[str, object]) -> str:
    lines: list[str] = []

    def add_section(title: str, items: object) -> None:
        lines.append(title)
        if isinstance(items, list) and items:
            for item in items:
                if isinstance(item, dict):
                    text = ", ".join(
                        f"{k}: {v}" for k, v in item.items() if v not in (None, "", [])
                    )
                    lines.append(f"- {text}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append("- None")
        lines.append("")

    add_section("Executive Summary", notes.get("executive_summary"))
    add_section("Decisions", notes.get("decisions"))
    add_section("Action Items", notes.get("action_items"))
    add_section("Risks / Blockers", notes.get("risks_blockers"))
    add_section("Follow-ups For Next Meeting", notes.get("follow_ups_next_meeting"))
    return "\n".join(lines).strip()


def _generate_notes(
    transcript_text: str,
    notes_in_english: bool,
    progress_callback: Callable[[str, int], None] | None = None,
) -> tuple[dict[str, object], str]:
    chunks = _split_text_for_notes(transcript_text, max_chars=max(NOTES_CHUNK_CHARS, 2000))
    if not chunks:
        empty = {
            "executive_summary": [],
            "decisions": [],
            "action_items": [],
            "risks_blockers": [],
            "follow_ups_next_meeting": [],
            "chunk_summaries": [],
        }
        return empty, _notes_json_to_text(empty)

    chunk_summaries: list[dict[str, object]] = []
    for i, chunk in enumerate(chunks, start=1):
        if progress_callback:
            prior_pct = int(((i - 1) / len(chunks)) * 90)
            progress_callback(f"Generating notes summary for chunk {i}/{len(chunks)}", prior_pct)
        chunk_summaries.append(_generate_chunk_summary(chunk, notes_in_english))
        if progress_callback:
            done_pct = int((i / len(chunks)) * 90)
            progress_callback(f"Generating notes summary for chunk {i}/{len(chunks)}", done_pct)

    if progress_callback:
        progress_callback("Generating final notes synthesis", 95)
    final_notes = _generate_final_notes(chunk_summaries, notes_in_english)
    if progress_callback:
        progress_callback("Generating final notes synthesis", 100)
    return final_notes, _notes_json_to_text(final_notes)


def _run_transcription_job(job_id: str) -> None:
    job = _get_job(job_id)
    if not job:
        return

    audio_path = str(job["audio_path"])
    task_mode = _normalize_task_mode(str(job.get("task_mode", "transcribe")))
    language = job["language"] if isinstance(job.get("language"), str) else None
    with_timestamps = bool(job.get("with_timestamps"))
    enable_speaker = bool(job.get("enable_speaker"))
    generate_notes = bool(job.get("generate_notes"))
    notes_in_english = bool(job.get("notes_in_english"))

    cleanup_paths = [audio_path]
    processed_path = audio_path
    source_for_transcription = audio_path
    force_chunk_conversion = False

    def update_detail(message: str) -> None:
        logger.info("job=%s %s", job_id, message)
        _set_job_fields(job_id, status_detail=message)
    _set_job_fields(job_id, status="running", status_detail="Started")

    try:
        probe_started = time.perf_counter()
        metadata = _probe_audio_metadata(audio_path)
        probe_elapsed = time.perf_counter() - probe_started
        logger.info("job=%s ffprobe metadata=%s elapsed=%.2fs", job_id, metadata, probe_elapsed)

        should_preprocess, preprocess_reason = _should_preprocess(
            audio_path,
            enable_speaker=enable_speaker,
            metadata=metadata,
        )
        logger.info("job=%s preprocess decision=%s reason=%s", job_id, should_preprocess, preprocess_reason)

        duration_seconds = metadata.get("duration")
        if not isinstance(duration_seconds, float):
            duration_seconds = None

        if should_preprocess and preprocess_reason == "speaker_diarization_enabled":
            _set_stage_state(job_id, "preprocess", status="running", progress=5, detail="Preprocessing audio")
            preprocess_started = time.perf_counter()
            processed_path, preprocess_warning = _preprocess_audio(
                audio_path,
                expected_seconds=duration_seconds,
                progress_callback=lambda pct: _set_stage_state(
                    job_id,
                    "preprocess",
                    status="running",
                    progress=pct,
                    detail=f"Preprocessing audio ({pct}%)",
                ),
            )
            logger.info("job=%s preprocess elapsed=%.2fs", job_id, time.perf_counter() - preprocess_started)
            if preprocess_warning:
                _append_job_warning(job_id, preprocess_warning)
            if processed_path != audio_path:
                cleanup_paths.append(processed_path)
            source_for_transcription = processed_path
            _set_stage_state(job_id, "preprocess", status="completed", progress=100, detail="Preprocessing complete")
        elif should_preprocess:
            force_chunk_conversion = True
            _set_stage_state(
                job_id,
                "preprocess",
                status="completed",
                progress=100,
                detail=f"Using per-chunk conversion ({preprocess_reason})",
            )
        else:
            _set_stage_state(
                job_id,
                "preprocess",
                status="skipped",
                progress=100,
                detail=f"Skipped preprocessing ({preprocess_reason})",
            )

        include_segments = with_timestamps or enable_speaker
        if enable_speaker and not with_timestamps:
            _append_job_warning(
                job_id,
                "Speaker attribution requires timing internally; timestamp display remains off in txt output.",
            )

        _set_stage_state(job_id, "transcribe", status="running", progress=1, detail="Transcribing audio")
        transcribe_started = time.perf_counter()
        transcript_text, transcript_segments = _transcribe_with_chunking(
            audio_path=source_for_transcription,
            task_mode=task_mode,
            language=language,
            include_segments=include_segments,
            duration_seconds=duration_seconds,
            force_conversion_chunks=force_chunk_conversion,
            progress_callback=lambda msg, pct: _set_stage_state(
                job_id,
                "transcribe",
                status="running",
                progress=pct,
                detail=msg,
            ),
        )
        logger.info("job=%s transcribe elapsed=%.2fs", job_id, time.perf_counter() - transcribe_started)
        _set_stage_state(job_id, "transcribe", status="completed", progress=100, detail="Transcription complete")
        if not transcript_text:
            _set_job_fields(
                job_id,
                status="failed",
                error="Transcription completed but returned empty text.",
                status_detail="Failed",
            )
            _set_stage_state(job_id, "finalize", status="failed", progress=100, detail="Failed")
            return

        if enable_speaker:
            _set_stage_state(job_id, "diarize", status="running", progress=10, detail="Running speaker attribution")
            try:
                speaker_turns = _run_diarization(processed_path)
                transcript_segments = _annotate_segments_with_speaker(transcript_segments, speaker_turns)
                _set_stage_state(
                    job_id,
                    "diarize",
                    status="completed",
                    progress=100,
                    detail="Speaker attribution complete",
                )
            except Exception as exc:  # noqa: BLE001
                _append_job_warning(job_id, f"Speaker attribution skipped: {exc}")
                _set_stage_state(
                    job_id,
                    "diarize",
                    status="warning",
                    progress=100,
                    detail="Speaker attribution skipped",
                )
        else:
            _set_stage_state(job_id, "diarize", status="skipped", progress=100)

        transcript_txt = _build_transcript_txt(
            transcript_text=transcript_text,
            segments=transcript_segments,
            with_timestamps=with_timestamps,
        )
        transcript_json = {
            "text": transcript_text,
            "segments": transcript_segments if include_segments else [],
        }

        notes_json = None
        notes_txt = None
        if generate_notes:
            _set_stage_state(job_id, "notes", status="running", progress=1, detail="Generating meeting notes")
            try:
                notes_json, notes_txt = _generate_notes(
                    transcript_text=transcript_text,
                    notes_in_english=notes_in_english,
                    progress_callback=lambda msg, pct: _set_stage_state(
                        job_id,
                        "notes",
                        status="running",
                        progress=pct,
                        detail=msg,
                    ),
                )
                _set_stage_state(job_id, "notes", status="completed", progress=100, detail="Meeting notes complete")
            except Exception as exc:  # noqa: BLE001
                _append_job_warning(job_id, f"Meeting notes generation failed: {exc}")
                _set_stage_state(job_id, "notes", status="warning", progress=100, detail="Meeting notes failed")
        else:
            _set_stage_state(job_id, "notes", status="skipped", progress=100)

        _set_stage_state(job_id, "finalize", status="running", progress=50, detail="Finalizing outputs")

        _set_job_fields(
            job_id,
            status="completed",
            status_detail="Completed",
            transcript_text=transcript_text,
            transcript_segments=transcript_segments,
            transcript_txt=transcript_txt,
            transcript_json=transcript_json,
            notes_json=notes_json,
            notes_txt=notes_txt,
            error=None,
        )
        _set_stage_state(job_id, "finalize", status="completed", progress=100, detail="Completed")
    except Exception as exc:  # noqa: BLE001
        logger.exception("job=%s failed", job_id)
        _set_job_fields(
            job_id,
            status="failed",
            status_detail="Failed",
            error=str(exc),
        )
        _set_stage_state(job_id, "finalize", status="failed", progress=100, detail="Failed")
    finally:
        for path in cleanup_paths:
            if path and os.path.exists(path):
                os.remove(path)
        _set_job_fields(job_id, audio_path=None)


def _session_job_ids() -> list[str]:
    raw_ids = session.get("job_ids", [])
    if not isinstance(raw_ids, list):
        return []
    return [jid for jid in raw_ids if isinstance(jid, str)]


def _remember_job_id(job_id: str) -> None:
    job_ids = _session_job_ids()
    if job_id not in job_ids:
        job_ids.append(job_id)
    session["job_ids"] = job_ids[-30:]


@app.get("/")
def index():
    selected_job_id = request.args.get("job_id")
    user_job_ids = _session_job_ids()
    if not selected_job_id and user_job_ids:
        selected_job_id = user_job_ids[-1]

    selected_job = _get_job(selected_job_id) if selected_job_id else None

    visible_jobs: list[dict[str, object]] = []
    for job_id in reversed(user_job_ids):
        job = _get_job(job_id)
        if not job:
            continue
        visible_jobs.append(
            {
                "id": job["id"],
                "status": job["status"],
                "status_detail": job.get("status_detail"),
                "overall_progress": job.get("overall_progress", 0),
                "original_filename": job["original_filename"],
                "task_mode": job.get("task_mode", "transcribe"),
                "created_at": _to_iso(job["created_at"]),  # type: ignore[arg-type]
            }
        )

    return render_template(
        "index.html",
        selected_job=selected_job,
        selected_job_id=selected_job_id,
        jobs=visible_jobs,
    )


@app.post("/transcribe")
def transcribe():
    if not os.getenv("OPENAI_API_KEY"):
        flash("Set OPENAI_API_KEY in your environment before transcribing.")
        return redirect(url_for("index"))

    upload = request.files.get("audio_file")
    if upload is None or upload.filename == "":
        flash("Choose an audio file first.")
        return redirect(url_for("index"))

    if not _is_allowed_file(upload.filename):
        flash("Unsupported file type. Please upload a common audio format.")
        return redirect(url_for("index"))

    filename = secure_filename(upload.filename)
    suffix = Path(filename).suffix
    task_mode = _normalize_task_mode(request.form.get("task_mode"))
    language = request.form.get("language", "").strip() or None
    with_timestamps = request.form.get("timestamps") == "on"
    enable_speaker = request.form.get("speaker_attribution") == "on"
    generate_notes = request.form.get("generate_notes") == "on"
    notes_in_english = request.form.get("notes_in_english") == "on"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            upload.save(tmp.name)
            temp_path = tmp.name

        job_id = _new_job(
            original_filename=filename,
            audio_path=temp_path,
            task_mode=task_mode,
            language=language,
            with_timestamps=with_timestamps,
            enable_speaker=enable_speaker,
            generate_notes=generate_notes,
            notes_in_english=notes_in_english,
        )
        _remember_job_id(job_id)
        logger.info("job=%s queued filename=%s mode=%s", job_id, filename, task_mode)
        executor.submit(_run_transcription_job, job_id)
        flash("Upload received. Processing started in background.")
        return redirect(url_for("index", job_id=job_id))
    except Exception as exc:  # noqa: BLE001
        flash(f"Transcription failed: {exc}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    return redirect(url_for("index"))


@app.get("/jobs/<job_id>/status")
def job_status(job_id: str):
    if job_id not in _session_job_ids():
        return jsonify({"error": "Job not found"}), 404

    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Job no longer available"}), 404

    return jsonify(
        {
            "id": job["id"],
            "status": job["status"],
            "status_detail": job.get("status_detail"),
            "overall_progress": job.get("overall_progress", 0),
            "stage_progress": job.get("stage_progress", {}),
            "stage_status": job.get("stage_status", {}),
            "stage_labels": job.get("stage_labels", STAGE_LABELS),
            "stage_order": job.get("stage_order", ["preprocess", "transcribe", "diarize", "notes", "finalize"]),
            "error": job.get("error"),
            "warnings": job.get("warnings", []),
            "has_transcript_txt": bool(job.get("transcript_txt")),
            "has_transcript_json": bool(job.get("transcript_json")),
            "has_notes_txt": bool(job.get("notes_txt")),
            "has_notes_json": bool(job.get("notes_json")),
        }
    )


@app.get("/jobs/<job_id>/download")
def legacy_download(job_id: str):
    return redirect(url_for("download_artifact", job_id=job_id, artifact="transcript_txt"))


@app.get("/jobs/<job_id>/download/<artifact>")
def download_artifact(job_id: str, artifact: str):
    if job_id not in _session_job_ids():
        flash("Job not found.")
        return redirect(url_for("index"))

    job = _get_job(job_id)
    if not job:
        flash("Job no longer exists.")
        return redirect(url_for("index"))

    artifact_map = {
        "transcript_txt": ("transcript_txt", "text/plain", f"meeting_transcript_{job_id[:8]}.txt"),
        "transcript_json": ("transcript_json", "application/json", f"meeting_transcript_{job_id[:8]}.json"),
        "notes_txt": ("notes_txt", "text/plain", f"meeting_notes_{job_id[:8]}.txt"),
        "notes_json": ("notes_json", "application/json", f"meeting_notes_{job_id[:8]}.json"),
    }
    if artifact not in artifact_map:
        flash("Unknown download artifact.")
        return redirect(url_for("index", job_id=job_id))

    key, mime, filename = artifact_map[artifact]
    payload = job.get(key)
    if payload in (None, "", []):
        flash("Artifact is not ready yet.")
        return redirect(url_for("index", job_id=job_id))

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix, mode="w", encoding="utf-8") as tmp:
        if isinstance(payload, str):
            tmp.write(payload)
        else:
            tmp.write(json.dumps(payload, ensure_ascii=False, indent=2))
        temp_name = tmp.name

    return send_file(
        temp_name,
        mimetype=mime,
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
