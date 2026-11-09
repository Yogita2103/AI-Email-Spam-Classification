"""
preprocess.py
-------------
ONE shared text-cleaning function used by BOTH training and prediction.
If training and prediction clean text differently, accuracy silently drops,
so never duplicate this logic anywhere else.
"""
import re

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_ADDR = re.compile(r"\S+@\S+")
_NUMBER = re.compile(r"\d+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Turn a raw email into clean lowercase text ready for TF-IDF.

    Steps: lowercase -> remove HTML -> mark URLs/email addresses/numbers
    -> remove punctuation -> collapse spaces.
    Stop-word removal is done later by TfidfVectorizer(stop_words="english").
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" urltoken ", text)          # keep the *fact* that a link existed
    text = _EMAIL_ADDR.sub(" emailtoken ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = _NON_LETTER.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    samples = [
        "WINNER!! Claim your FREE $1000 prize at http://win-money.com NOW",
        "<html><b>Hi John</b>, meeting moved to 3pm. Mail me at john@corp.com</html>",
        "",
    ]
    for s in samples:
        print("RAW  :", s)
        print("CLEAN:", clean_text(s))
        print()
