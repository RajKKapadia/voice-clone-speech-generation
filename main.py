import argparse
import torch
import soundfile as sf
import pickle
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PROMPT_PATH = "my_voice_prompt.pkl"


def load_model():
    return Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def clone(args):
    model = load_model()

    with open(args.transcript, "r") as f:
        ref_text = f.read().strip()

    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=args.audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    with open(args.output, "wb") as f:
        pickle.dump(voice_prompt, f)

    print(f"Saved voice prompt to: {args.output}")


def synth(args):
    model = load_model()

    with open(args.prompt, "rb") as f:
        voice_prompt = pickle.load(f)

    text = args.text.strip()
    if text and text[-1] not in ".!?":
        text += "."

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=voice_prompt,
        max_new_tokens=4096,
        top_p=0.85,
        temperature=0.7,
    )
    sf.write(args.output, wavs[0], sr)
    print(f"Saved: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Voice cloning and speech generation")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- clone subcommand ---
    clone_p = sub.add_parser("clone", help="Create a voice clone prompt from reference audio")
    clone_p.add_argument("--audio", required=True, help="Path to reference audio file (e.g. untitled.wav)")
    clone_p.add_argument("--transcript", required=True, help="Path to transcript text file for the audio")
    clone_p.add_argument("--output", default=PROMPT_PATH, help=f"Where to save the voice prompt (default: {PROMPT_PATH})")

    # --- synth subcommand ---
    synth_p = sub.add_parser("synth", help="Generate speech using a saved voice clone prompt")
    synth_p.add_argument("--text", required=True, help="Text to speak")
    synth_p.add_argument("--output", default="output.wav", help="Output wav file path (default: output.wav)")
    synth_p.add_argument("--prompt", default=PROMPT_PATH, help=f"Path to saved voice prompt (default: {PROMPT_PATH})")

    args = parser.parse_args()

    if args.command == "clone":
        clone(args)
    elif args.command == "synth":
        synth(args)


if __name__ == "__main__":
    main()
