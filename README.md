# Talkit — push-to-talk dictation for macOS

Hold **Right Option**, speak, release → your words are transcribed locally (Whisper on Apple Silicon via MLX), optionally cleaned up by Claude, and pasted into whatever app is focused.

| Action | Gesture |
|---|---|
| Push-to-talk | Hold Right ⌥ |
| Hands-free | Double-tap Right ⌥, tap once more to finish |
| Cancel | Esc while recording |

Menu bar: 🎙 idle · 🔴 recording · 🟠 hands-free · ⏳ transcribing. The menu has an AI-cleanup toggle and your recent transcripts (click one to copy it). History is written to `~/.local/share/talkit/history.jsonl`.

## The Talkit window

The window opens at launch. Closing it leaves Talkit running in the menu bar; reopen it from 🎙 → **Open Talkit Window**. Start with `--no-window` to begin in the menu bar only.

- **Sidebar:** status, a **Start/Stop transcript** button with a live mic level meter, the AI cleanup switch, and every transcript session (the live one has a green dot).
- **Transcripts:** entries appear as you speak, with "Hearing speech…" and "Transcribing…" indicators. Search (⌘F), **Copy** all text, **Reveal** the file in Finder, or **Export** as Markdown, plain text, SRT subtitles or JSON. Hover over a line to copy just that line.
- **Dictations:** everything you've dictated, the app you dictated into, and what Whisper heard before cleanup. You can search, copy and export these too.
- **Activity:** the log that used to go to the terminal.

## Continuous transcript mode

No key holding: the mic stays open, speech is detected automatically, and each utterance is transcribed and appended to a timestamped Markdown file in `~/Documents/Talkit Transcripts/`.

- Start it from the menu bar (**Continuous transcript**), or launch with `.venv/bin/python talkit.py --continuous`.
- The icon turns 🟢 while it's running. Click the same menu item to stop. **Open transcripts folder** opens the files.
- Each session gets its own file, e.g. `2026-11-06_16-12-10.md`, with lines like `**[16:12:31]** Next week we will move production traffic over.`
- Push-to-talk still works at the same time and gets priority, though anything you dictate is also written to the transcript.
- If it misses quiet speech, lower `TALKIT_VAD_THRESHOLD` (default `0.008`). If it picks up background noise, raise it.

## Setup

Needs roughly 3 GB free: about 1 GB for dependencies and 1.6 GB for the default Whisper model.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional, turns on AI cleanup
.venv/bin/python talkit.py
```

The first run downloads the Whisper model. macOS will then ask for permissions for your terminal app (System Settings → Privacy & Security):
- **Microphone**: to record
- **Input Monitoring**: to see the hotkey
- **Accessibility**: to paste with ⌘V

After granting them, restart the terminal and run the app again.

## Configuration (env vars)

| Var | Default | Notes |
|---|---|---|
| `TALKIT_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Use `mlx-community/whisper-small-mlx` for a smaller, faster model (~500 MB) |
| `TALKIT_LANGUAGE` | auto | e.g. `en`; setting it is slightly faster and more accurate |
| `TALKIT_CLEANUP` | `auto` | `on` / `off` / `auto` (on when an API key is set) |
| `TALKIT_CLEANUP_MODEL` | `claude-opus-5-5` | Any Claude model ID, e.g. `claude-haiku-5-5` for lower latency and cost |
| `TALKIT_VOCAB` | — | Comma-separated names and jargon that help Whisper spell them right |
| `TALKIT_TRANSCRIPT_DIR` | `~/Documents/Talkit Transcripts` | Where continuous-mode transcripts go |
| `TALKIT_VAD_THRESHOLD` | `0.008` | Speech-detection sensitivity for continuous mode |
| `TALKIT_HOTKEY_KEYCODE` | `61` (Right ⌥) | macOS virtual keycode of a modifier key |

AI cleanup removes filler words, applies self-corrections ("at 3, actually 4" → "at 4"), fixes punctuation, turns spoken "new line" / "bullet point" into formatting, and adjusts tone to the destination app. If the API call fails, the raw transcript is pasted instead.
