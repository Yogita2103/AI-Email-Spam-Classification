"""
build_dataset.py
----------------
Reads the extracted SpamAssassin folders inside data/raw/ and creates
data/emails.csv with two columns: label (spam / ham) and text.

Expected layout after extracting the .tar.bz2 files into data/raw/:
    data/raw/easy_ham/  easy_ham_2/  hard_ham/  spam/  spam_2/

Run from the project root:  python src/build_dataset.py
"""
import re
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "emails.csv"

# folder name -> label
FOLDERS = {
    "easy_ham": "ham",
    "easy_ham_2": "ham",
    "hard_ham": "ham",
    "spam": "spam",
    "spam_2": "spam",
}
MAX_CHARS = 20000
_TAGS = re.compile(r"<[^>]+>")


def _get_body(msg) -> str:
    """Return the readable body: plain text if available, otherwise stripped HTML."""
    parts = []
    html_parts = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        try:
            content = part.get_content()
        except Exception:  # broken encodings are common in old spam
            payload = part.get_payload(decode=True) or b""
            content = payload.decode("latin-1", errors="ignore")
        if not isinstance(content, str):
            continue
        if ctype == "text/plain":
            parts.append(content)
        elif ctype == "text/html":
            html_parts.append(_TAGS.sub(" ", content))
    return "\n".join(parts if parts else html_parts)


def parse_email_file(path: Path) -> str:
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    subject = str(msg.get("subject", "") or "")
    return (subject + "\n" + _get_body(msg))[:MAX_CHARS]


def main() -> None:
    rows = []
    for folder, label in FOLDERS.items():
        folder_path = RAW_DIR / folder
        if not folder_path.exists():
            print(f"[skip] {folder_path} not found")
            continue
        count = 0
        for path in folder_path.iterdir():
            if not path.is_file() or path.name == "cmds" or path.name.startswith("."):
                continue
            try:
                text = parse_email_file(path)
            except Exception as exc:
                print(f"[warn] could not read {path.name}: {exc}")
                continue
            if text.strip():
                rows.append({"label": label, "text": text})
                count += 1
        print(f"[ok]   {folder:<11} -> {count} emails ({label})")

    if not rows:
        sys.exit("No emails found. Extract the dataset archives into data/raw/ first.")

    df = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)
    print("\nSaved:", OUT_FILE)
    print("Total emails:", len(df))
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
