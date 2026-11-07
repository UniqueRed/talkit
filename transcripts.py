"""Reading transcript session files and exporting them in other formats."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

LINE = re.compile(r"^\*\*\[(\d{2}:\d{2}:\d{2})\]\*\*\s?(.*)$")
EXPORT_FORMATS = {"md": "Markdown", "txt": "Plain text", "json": "JSON", "srt": "SRT subtitles"}


def session_start(path: Path) -> datetime:
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime)


def session_title(path: Path) -> str:
    return session_start(path).strftime("%a %b %-d, %-I:%M %p")


def read_entries(path: Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return [{"time": m[1], "text": m[2]} for m in map(LINE.match, lines) if m]


def session_info(path: Path) -> dict:
    entries = read_entries(path)
    return {
        "id": path.stem,
        "title": session_title(path),
        "count": len(entries),
        "words": sum(len(e["text"].split()) for e in entries),
    }


def list_sessions(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    return [session_info(p) for p in sorted(directory.glob("*.md"), reverse=True)]


def _absolute_times(start: datetime, entries: list[dict]) -> list[datetime]:
    """Entry stamps are clock times; turn them into datetimes, handling midnight rollover."""
    out, day, prev = [], start.date(), None
    for e in entries:
        t = datetime.strptime(e["time"], "%H:%M:%S").time()
        if prev is not None and t < prev:
            day += timedelta(days=1)
        prev = t
        out.append(datetime.combine(day, t))
    return out


def _srt_ts(td: timedelta) -> str:
    ms = max(0, int(td.total_seconds() * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_session(path: Path, fmt: str) -> str:
    entries = read_entries(path)
    start = session_start(path)
    title = session_title(path)

    if fmt == "md":
        body = "".join(f"**[{e['time']}]** {e['text']}\n\n" for e in entries)
        return f"# Transcript — {title}\n\n{body}"
    if fmt == "txt":
        body = "".join(f"[{e['time']}] {e['text']}\n" for e in entries)
        return f"Transcript — {title}\n\n{body}"
    if fmt == "json":
        times = _absolute_times(start, entries)
        return json.dumps({
            "session": path.stem,
            "started": start.isoformat(),
            "entries": [{"time": t.isoformat(), "text": e["text"]} for t, e in zip(times, entries)],
        }, indent=2, ensure_ascii=False) + "\n"
    if fmt == "srt":
        times = _absolute_times(start, entries)
        base = min([start, *times]) if times else start
        blocks = []
        for i, (t, e) in enumerate(zip(times, entries)):
            begin = t - base
            spoken = timedelta(seconds=max(1.5, len(e["text"].split()) * 0.4))
            end = begin + spoken
            if i + 1 < len(times):
                end = min(end, times[i + 1] - base - timedelta(milliseconds=100))
            end = max(end, begin + timedelta(milliseconds=500))
            blocks.append(f"{i + 1}\n{_srt_ts(begin)} --> {_srt_ts(end)}\n{e['text']}\n")
        return "\n".join(blocks)
    raise ValueError(f"unknown format {fmt!r}")


def export_dictations(items: list[dict], fmt: str) -> str:
    """items: [{ts, app, raw, text}] oldest-first."""
    if fmt == "json":
        return json.dumps(items, indent=2, ensure_ascii=False) + "\n"
    lines = []
    for it in items:
        stamp = datetime.fromtimestamp(it["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        app = f" · {it['app']}" if it.get("app") else ""
        lines.append(f"**[{stamp}{app}]** {it['text']}\n\n" if fmt == "md" else f"[{stamp}{app}] {it['text']}\n")
    head = "# Dictation history\n\n" if fmt == "md" else "Dictation history\n\n"
    return head + "".join(lines)
