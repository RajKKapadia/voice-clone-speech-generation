# Voice Clone & Speech Generation

Clone any voice from a short audio sample, then generate unlimited speech in that voice.

Built on [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

## Requirements

- Python 3.12+
- CUDA-capable GPU
- Dependencies: `torch`, `soundfile`, `qwen_tts`

## Quick Start

### Step 1 — Clone a voice

Provide a reference audio file and its transcript to build a reusable voice prompt:

```bash
python main.py clone --audio untitled.wav --transcript untitled.txt
```

This saves the voice prompt to `my_voice_prompt.pkl`.

To save it somewhere else:

```bash
python main.py clone --audio untitled.wav --transcript untitled.txt --output custom_prompt.pkl
```

### Step 2 — Generate speech

Use the saved voice prompt to synthesize new speech from any text:

```bash
python main.py synth --text "Hello, this is my cloned voice speaking."
```

This writes `output.wav` in the current directory.

To customize the output path or use a different prompt file:

```bash
python main.py synth --text "Another sentence." --output speech.wav --prompt custom_prompt.pkl
```

## CLI Reference

```
usage: main.py {clone,synth} ...

Voice cloning and speech generation
```

### `clone`

| Flag             | Required | Default                | Description                          |
| ---------------- | -------- | ---------------------- | ------------------------------------ |
| `--audio`        | yes      | —                      | Path to reference audio file (.wav)  |
| `--transcript`   | yes      | —                      | Path to transcript of the audio      |
| `--output`       | no       | `my_voice_prompt.pkl`  | Where to save the voice prompt       |

### `synth`

| Flag       | Required | Default                | Description                          |
| ---------- | -------- | ---------------------- | ------------------------------------ |
| `--text`   | yes      | —                      | Text to speak                        |
| `--output` | no       | `output.wav`           | Output wav file path                 |
| `--prompt` | no       | `my_voice_prompt.pkl`  | Path to a saved voice prompt         |

## How It Works

1. **Clone** — The model encodes your reference audio + transcript into a voice prompt (speaker embedding) and serializes it to a `.pkl` file.
2. **Synth** — The model loads that voice prompt and generates new waveforms for arbitrary text, preserving the cloned voice characteristics.

The voice prompt only needs to be created once per speaker. After that, you can run `synth` as many times as you want without re-processing the reference audio.
