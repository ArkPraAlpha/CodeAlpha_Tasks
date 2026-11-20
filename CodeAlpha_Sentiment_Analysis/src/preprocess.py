# Text cleaning utilities.

# Two versions of the text are kept:
#   * text_clean - light cleaning only (keeps capitals, punctuation, emojis) because
#                  VADER and transformers use those signals.
#   * text_ml    - lower-cased letters only, used by the TF-IDF machine-learning model.
# 
import html
import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(text):
    # Remove HTML, URLs and @mentions; keep the words of hashtags.# 
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", "")
    return WS_RE.sub(" ", text).strip()


def normalize_for_ml(text):
    # Lower-case and keep only letters/apostrophes (stop-words are kept on
    # purpose, because words like 'not' flip the meaning of a sentence). 
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return WS_RE.sub(" ", text).strip()


def add_text_columns(df, text_col="text"):
    df = df.copy()
    df["text_clean"] = df[text_col].map(clean_text)
    df["text_ml"] = df[text_col].map(normalize_for_ml)
    df["word_count"] = df["text_clean"].str.split().str.len()
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    return df
