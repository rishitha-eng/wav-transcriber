"""Local web UI for the wav_transcriber pipeline (Gradio) — minimal version:
drop in a WAV file, optionally pick a target language, get the customer's
speech back as numbered, meaningfully split sentences in a single box."""

from __future__ import annotations

import gradio as gr

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


def run_pipeline(audio_path, target_language_label):
    if not audio_path:
        return ""

    try:
        result = transcribe_customer(audio_path, translate_to=LANGUAGE_OPTIONS.get(target_language_label))
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
        return f"Error: {e}"

    if result.customer_sentences:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(result.customer_sentences, start=1))
        if result.used_second_speaker_heuristic:
            numbered += "\n\n(Two speakers found, no channel info — assumed the 2nd to speak is the customer.)"
        return numbered

    if len(result.human_speakers) > 1:
        return (
            "Found more than one human speaker in this file and couldn't tell them apart "
            "automatically."
        )
    return "No customer speech detected."


with gr.Blocks(title="Customer Voice to Text") as demo:
    gr.Markdown("# Customer Voice → Text")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Drop audio here", sources=["upload"])
        with gr.Column():
            target_language = gr.Dropdown(
                list(LANGUAGE_OPTIONS.keys()), value="Original (no translation)", label="Translate to"
            )
            sentences_output = gr.Textbox(label="Customer speech", lines=18)

    inputs = [audio_input, target_language]
    audio_input.change(run_pipeline, inputs=inputs, outputs=sentences_output)
    target_language.change(run_pipeline, inputs=inputs, outputs=sentences_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
