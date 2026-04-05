"""Gradio web UI for Rocky Voice Synthesizer."""

import tempfile
import os

import gradio as gr

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rocky import VoiceSynthesizer, VoiceConfig


# Pre-create synthesizer instance to avoid re-initialization
_synth: VoiceSynthesizer | None = None


def get_synth(sample_rate: int) -> VoiceSynthesizer:
    global _synth
    if _synth is None:
        config = VoiceConfig(sample_rate=sample_rate)
        _synth = VoiceSynthesizer(config)
    return _synth


def synthesize_voice(text: str, sample_rate: int):
    if not text or not text.strip():
        return None, "Please enter some text."

    text = text.strip()
    synth = get_synth(sample_rate)

    audio = synth.generate(text)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    import soundfile as sf
    sf.write(tmp_path, audio, sample_rate)

    msg = f"Generated '{text}' as Rocky voice ({len(audio) / sample_rate:.1f}s, {sample_rate}Hz)"
    return tmp_path, msg


def consistency_test(text: str, sample_rate: int):
    """Verify that the same text always produces identical output."""
    if not text or not text.strip():
        return "Please enter some text."

    text = text.strip()
    synth = get_synth(sample_rate)

    audio1 = synth.generate(text)
    audio2 = synth.generate(text)

    if audio1.shape != audio2.shape:
        return f"FAIL: Shapes differ — {audio1.shape} vs {audio2.shape}"

    diff = float(abs(audio1 - audio2).max())
    if diff < 1e-6:
        return f"PASS: '{text}' produced identical output (max diff: {diff:.2e})"
    else:
        return f"FAIL: Outputs differ (max diff: {diff:.2e})"


with gr.Blocks(title="Rocky Voice Synthesizer") as demo:
    gr.Markdown("# Rocky Voice Synthesizer")
    gr.Markdown(
        "Synthesize Rocky (Eridian alien) voice from text. "
        "Rocky communicates by tapping its body to produce resonant sounds. "
        "Same text always produces identical output — guaranteed by deterministic hashing."
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter any text in Chinese, English, or any language...",
                lines=3,
            )
        with gr.Column(scale=1):
            sample_rate_input = gr.Dropdown(
                choices=[16000, 22050, 44100],
                value=22050,
                label="Sample Rate (Hz)",
            )

    with gr.Row():
        generate_btn = gr.Button("Synthesize", variant="primary")
        consistency_btn = gr.Button("Test Consistency", variant="secondary")

    with gr.Row():
        audio_output = gr.Audio(label="Voice Output")
        status_output = gr.Textbox(label="Status", lines=2)

    gr.Markdown("### About Rocky")
    gr.Markdown(
        "Rocky is an Eridian alien from the novel **Project Hail Mary** by Andy Weir. "
        "Eridians communicate by tapping their bodies to produce resonant, bell-like sounds. "
        "| Characteristic | Value |\n"
        "|----------------|-------|\n"
        "| Base frequency | 40-80 Hz |\n"
        "| Sound quality | Deep, resonant, bell/gong-like |\n"
        "| Tempo | Slow, deliberate tapping |\n"
    )

    generate_btn.click(
        fn=synthesize_voice,
        inputs=[text_input, sample_rate_input],
        outputs=[audio_output, status_output],
    )
    consistency_btn.click(
        fn=consistency_test,
        inputs=[text_input, sample_rate_input],
        outputs=[status_output],
    )

    gr.Markdown(
        "\n---\n*Same text → same audio every time. Different text → completely different sounds.*"
    )


def launch(port: int = 7860):
    demo.launch(server_port=port, server_name="127.0.0.1", inbrowser=True, theme=gr.themes.Soft(), allowed_paths=[tempfile.gettempdir()])


if __name__ == "__main__":
    launch()
