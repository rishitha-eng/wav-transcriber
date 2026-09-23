# wav-transcriber

Extract a customer's speech from a call recording (WAV), transcribed and split into
meaningful sentences — with optional translation, and single-turn audio chunks
exported for STT benchmarking ground truth.

## How it works

Speakers are separated one of two ways, picked automatically per file:

- **Channel split** (no setup needed): if the WAV is stereo with two genuinely
  different channels — the common call-recording layout, one speaker per channel —
  each channel is transcribed independently. The customer is picked by a fixed
  channel convention (`--customer-channel`, default `LEFT`).
- **Diarization** (needs a Hugging Face token): for mono or genuinely mixed-down
  audio, [pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1)
  figures out who spoke when, and a voice classifier tells human from AI/bot voices
  to auto-pick the customer.

Transcription runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(multilingual, auto-detects language). Translation (optional) runs locally via
[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For files that need diarization, accept the model terms at
https://huggingface.co/pyannote/speaker-diarization-community-1, then set:

```bash
export HF_TOKEN=hf_your_token_here
```

## Usage

**Web UI** — drop a file in, get sentences back:

```bash
python app.py
# open http://localhost:7860
```

**CLI** — single file or a whole folder:

```bash
python run.py /path/to/call.wav
python run.py /path/to/folder --out-dir /path/to/results
```

Run `python run.py --help` for all options (target language, AI-voice threshold,
manual speaker override, chunk export, etc.).

## Tests

```bash
python -m pytest tests/
```

## Ingestion pipeline (optional)

`ingestion/` has a separate OpenSearch → Azure Storage → ground-truth pipeline for
bulk-sampling call recordings for STT benchmarking. See `ingestion/config.example.json`
— copy to `config.json`, fill in real endpoint/auth details, then run
`python -m ingestion.run_pipeline`. Needs network access to the relevant internal
systems; not required for the transcriber itself.
