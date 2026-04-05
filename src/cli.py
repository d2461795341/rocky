"""CLI interface for Rocky Voice Synthesizer."""

import argparse
import os
import sys
import tempfile


def main():
    from src.rocky import VoiceConfig, VoiceSynthesizer

    parser = argparse.ArgumentParser(
        prog="rocky-synth",
        description="Synthesize Rocky (Eridian) alien voice from text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rocky-synth "你好 Rocky"
  rocky-synth "Hello Rocky" --sample-rate 44100 -o output.wav
        """,
    )
    parser.add_argument("text", nargs="*", help="Text to synthesize into voice")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output .wav file path (default: prints path to stdout)")
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Audio sample rate in Hz (default: 22050)")

    args = parser.parse_args()

    if not args.text:
        parser.error("the following arguments are required: text")

    text = " ".join(args.text)

    config = VoiceConfig(sample_rate=args.sample_rate)
    synth = VoiceSynthesizer(config)

    if args.output:
        output_path = args.output
    else:
        safe_name = "".join(c if c.isalnum() else "_" for c in text[:20])
        output_path = os.path.join(tempfile.gettempdir(), f"rocky_{safe_name}.wav")

    try:
        result = synth.synthesize_file(text, output_path)
        if args.output:
            print(f"Saved: {result}")
        else:
            print(result)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
