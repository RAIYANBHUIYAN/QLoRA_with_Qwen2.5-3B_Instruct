"""Convert statute raw text in ./raw into QLoRA messages JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
OUT_PATH = ROOT / "data" / "land_law_sft.jsonl"

ACTS = {
    "TPA_1882_raw.txt": {
        "name": "the Transfer of Property Act, 1882",
        "short": "TPA 1882",
    },
    "SAT_1950_raw.txt": {
        "name": "the State Acquisition and Tenancy Act, 1950",
        "short": "SAT 1950",
    },
    "NAT_1949_raw.txt": {
        "name": "the Non-Agricultural Tenancy Act, 1949",
        "short": "NAT 1949",
    },
    "Land_Tax_2023_raw.txt": {
        "name": "the Land Development Tax Ordinance, 1976",
        "short": "Land Development Tax Ordinance 1976",
    },
    "ARIPA_2017_raw.txt": {
        "name": "the Acquisition and Requisition of Immovable Property Ordinance, 1982",
        "short": "ARIPO 1982",
    },
}

START_RE = re.compile(
    r"(This Act may be called|This Ordinance may be called)",
    re.I,
)
JUNK_LINE_RE = re.compile(
    r"(bdlaws\.minlaw\.gov\.bd|act-print-\d+|^\s*\d+/\d+\s*$|CONTENTS|SECTIONS)",
    re.I,
)
SECTION_RE = re.compile(
    r"(?:(?<=\n)|(?<=^))"
    r"(?:(?P<head>[A-Za-z][^\n]{0,80}?)\s+)?"
    r"(?P<num>\d{1,3}[A-Za-z]?)\.\s+"
    r"(?=(?:\([0-9]+\)\s+)?[A-Z\"“(])"
)


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if JUNK_LINE_RE.search(line):
            continue
        if re.fullmatch(r"\s*[A-Za-z]\s*", line):
            continue
        line = re.sub(r"\[\d+[A-Za-z]?\]", "", line)
        line = re.sub(r"[*\[\]<>]+", " ", line)
        lines.append(line)
    text = " ".join(" ".join(lines).split())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sections(text: str) -> list[dict]:
    matches = list(SECTION_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        num = m.group("num")
        if num.isdigit() and int(num) > 400:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = clean_text(text[start:end])
        head = clean_text(m.group("head") or "")
        if len(body) < 40:
            continue
        if len(body) > 2500:
            body = body[:2500].rsplit(" ", 1)[0] + "..."
        sections.append({"num": num, "title": head, "text": body})
    return sections


def examples_for_section(act: dict, sec: dict) -> list[dict]:
    cite = f"Section {sec['num']} of {act['name']}"
    title = sec["title"]
    body = sec["text"]
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"What does {cite} say?",
                },
                {
                    "role": "assistant",
                    "content": f"{cite} provides: {body}",
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Quote {act['short']} section {sec['num']}.",
                },
                {
                    "role": "assistant",
                    "content": f"{act['short']}, section {sec['num']}: {body}",
                },
            ]
        },
    ]
    if title and len(title) > 3:
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Under {act['short']}, explain the provision on {title}.",
                    },
                    {
                        "role": "assistant",
                        "content": f"That is covered by {cite}. {body}",
                    },
                ]
            }
        )
    return rows


def convert_file(path: Path, meta: dict) -> list[dict]:
    data = path.read_bytes()
    raw = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raw = data.decode("utf-8", errors="ignore")
    raw = (
        raw.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\ufffd", '"')
    )
    m = START_RE.search(raw)
    raw = raw[m.start() :] if m else raw
    rows = []
    for sec in split_sections(raw):
        rows.extend(examples_for_section(meta, sec))
    return rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for filename, meta in ACTS.items():
        path = RAW_DIR / filename
        if not path.exists():
            print("missing", path)
            continue
        rows = convert_file(path, meta)
        print(f"{filename}: {len(rows)} examples")
        all_rows.extend(rows)

    seen = set()
    unique = []
    for row in all_rows:
        key = row["messages"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(unique)} rows to {OUT_PATH}")
    print("sample:", json.dumps(unique[0], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
