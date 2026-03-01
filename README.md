# Meeting Transcriber (OpenAI Whisper)

A Python Flask app with a lightweight web UI for long meeting recordings (including mixed-language conversations).

## Features

- Upload common audio/video formats and run as background jobs
- Mode toggle:
  - `Transcribe` (keep original language; language hint optional)
  - `Translate` (English output)
- Chunking for large uploads (works around OpenAI file-size request limits)
- Optional segment timestamps (default ON)
- Transcript artifacts:
  - `transcript.txt` (human-readable)
  - `transcript.json` (machine-readable segments)
- Optional speaker attribution (beta, feature-flagged)
- Optional meeting notes generation with chunked summarization
- Optional `Notes in English` while keeping transcript in original language

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Required:

- `OPENAI_API_KEY`

Common optional settings:

- `WHISPER_MODEL` (default `whisper-1`)
- `NOTES_MODEL` (default `gpt-5.2`)
- `MAX_UPLOAD_MB` (app upload limit; default `250`)
- `JOB_WORKERS` (parallel background jobs; default `2`)
- `OPENAI_MAX_BYTES` (OpenAI request cap; default `26214400`)
- `CHUNK_MINUTES` (audio split window; default `10`)
- `CHUNK_TRANSCRIBE_WORKERS` (parallel chunk transcriptions; default `3`)
- `NOTES_CHUNK_CHARS` (transcript chars per notes chunk; default `12000`)
- `LOG_LEVEL` (default `INFO`)

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Optional Speaker Attribution (Beta)

Speaker attribution is disabled by default.

1. Set `ENABLE_DIARIZATION_BETA=true` in `.env`
2. Set `HF_TOKEN` in `.env`
3. Install optional diarization dependencies (not required for base app):
   - `pyannote.audio` and its runtime requirements

If diarization is requested but unavailable, the job still completes and shows a warning.

## Notes

- Jobs are stored in-memory and tied to browser session IDs.
- Restarting the app clears in-memory job state/history.
- Smart pre-processing uses `ffprobe` metadata and skips conversion for already-supported inputs.
- Pre-processing is forced when speaker diarization is enabled.
- Chunk production and chunk transcription run in a pipelined/parallel flow for faster long-job throughput.
