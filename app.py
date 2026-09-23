"""Local web UI for the wav_transcriber pipeline (Gradio) — minimal version:
drop in a WAV file, optionally pick a target language, get the customer's
speech back as numbered, meaningfully split sentences in a single box, plus
a .docx download of the same."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
from docx import Document

from wav_transcriber.pipeline import transcribe_customer

LANGUAGE_OPTIONS = {
    "Original (no translation)": None,
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "Arabic": "ar",
    "Portuguese": "pt",
    "German": "de",
    "Indonesian": "id",
    "Filipino / Tagalog": "tl",
    "Bengali": "bn",
    "Urdu": "ur",
    "Chinese": "zh",
    "Japanese": "ja",
    "Russian": "ru",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Vietnamese": "vi",
    "Thai": "th",
    "Swahili": "sw",
}


def export_docx(sentences: list[str], source_name: str) -> str:
    doc = Document()
    doc.add_heading("Customer Speech", level=1)
    doc.add_paragraph(f"Source: {source_name}")
    for i, sentence in enumerate(sentences, start=1):
        doc.add_paragraph(f"{i}. {sentence}")

    out_dir = Path(tempfile.mkdtemp(prefix="wav_transcriber_"))
    out_path = out_dir / f"{Path(source_name).stem}_customer_speech.docx"
    doc.save(out_path)
    return str(out_path)


def run_pipeline(audio_path, target_language_label):
    if not audio_path:
        return "", gr.update(value=None, visible=False)

    try:
        result = transcribe_customer(audio_path, translate_to=LANGUAGE_OPTIONS.get(target_language_label))
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
        return f"Error: {e}", gr.update(value=None, visible=False)

    if result.customer_sentences:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(result.customer_sentences, start=1))
        if result.used_second_speaker_heuristic:
            numbered += "\n\n(Two speakers found, no channel info — assumed the 2nd to speak is the customer.)"
        docx_path = export_docx(result.customer_sentences, Path(audio_path).name)
        return numbered, gr.update(value=docx_path, visible=True)

    if len(result.human_speakers) > 1:
        return (
            "Found more than one human speaker in this file and couldn't tell them apart "
            "automatically."
        ), gr.update(value=None, visible=False)
    return "No customer speech detected.", gr.update(value=None, visible=False)


with gr.Blocks(title="Customer Voice to Text") as demo:
    gr.Markdown("# Customer Voice → Text")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Drop audio here", sources=["upload"])
        with gr.Column():
            target_language = gr.Dropdown(
                list(LANGUAGE_OPTIONS.keys()), value="Original (no translation)", label="Translate to"
            )
            sentences_output = gr.Textbox(label="Customer speech", lines=16)
            download_btn = gr.DownloadButton("Download as .docx", visible=False)

    inputs = [audio_input, target_language]
    outputs = [sentences_output, download_btn]
    audio_input.change(run_pipeline, inputs=inputs, outputs=outputs)
    target_language.change(run_pipeline, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
