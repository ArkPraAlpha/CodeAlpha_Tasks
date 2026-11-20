"""Emotion detection with a lexicon (Plutchik-style emotions, as in the NRC lexicon).

A compact built-in lexicon is included so the project works offline. For better
coverage, download the NRC Emotion Lexicon (word-level file) and pass its path
with --nrc-lexicon; its entries are merged into the built-in lexicon.
"""
import re
from collections import defaultdict

import pandas as pd

EMOTIONS = ["joy", "trust", "anticipation", "surprise", "sadness", "fear", "anger", "disgust"]

_BUILTIN = {
    "joy": "love loved loves loving happy happily delighted delight great excellent amazing awesome "
           "wonderful fantastic perfect pleased enjoy enjoyed fun glad thrilled brilliant superb "
           "gorgeous beautiful impressive impressed smooth",
    "trust": "reliable trust trusted dependable durable honest solid recommend recommended safe "
             "secure consistent genuine authentic helpful friendly clear secure praise",
    "anticipation": "hope hoping excited expect expecting await soon eager anticipate waiting "
                    "upcoming looking planning",
    "surprise": "surprised surprising unexpected shocked amazed astonished suddenly wow unbelievable "
                "stunned unexpectedly",
    "sadness": "sad disappointed disappointing disappointment unhappy regret sorry unfortunately "
               "miserable upset depressing poor worthless heartbroken waited late delayed",
    "fear": "afraid fear scared worried worry worrying concern concerned concerns risk risky dangerous "
            "unsafe nervous anxious terrified threat backlash",
    "anger": "angry furious annoyed annoying outraged rage hate hated terrible awful horrible "
             "ridiculous unacceptable useless mad irritating frustrating frustrated rude refused "
             "cheated",
    "disgust": "disgusting gross filthy nasty revolting rotten junk garbage trash broken defective "
               "faulty scam fake flimsy cheap ugly damaged torn crushed",
}

LEXICON = {}
for _emotion, _words in _BUILTIN.items():
    for _w in _words.split():
        LEXICON.setdefault(_w, set()).add(_emotion)

NEGATORS = {"not", "no", "never", "n't", "hardly", "without", "isn't", "wasn't", "don't",
            "didn't", "doesn't", "cannot", "can't", "won't"}
NEGATION_FLIP = {"joy": "sadness", "sadness": "joy"}  # "not happy" ~ sadness
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def load_nrc_lexicon(path):
    """Merge the NRC word-level lexicon (word <TAB> emotion <TAB> 0/1) into LEXICON."""
    added = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) == 3 and parts[2] == "1" and parts[1] in EMOTIONS:
                LEXICON.setdefault(parts[0], set()).add(parts[1])
                added += 1
    print(f"  Loaded {added} NRC emotion entries")


def emotion_counts(text):
    counts = defaultdict(float)
    tokens = TOKEN_RE.findall(text.lower())
    for i, tok in enumerate(tokens):
        if tok not in LEXICON:
            continue
        negated = any(t in NEGATORS for t in tokens[max(0, i - 3):i])
        for emotion in LEXICON[tok]:
            if negated:
                flipped = NEGATION_FLIP.get(emotion)
                if flipped:
                    counts[flipped] += 1
            else:
                counts[emotion] += 1
    return counts


def detect_emotions(texts):
    """Return a DataFrame with one column per emotion (word counts) plus 'dominant_emotion'."""
    rows = [emotion_counts(t) for t in texts]
    df = pd.DataFrame(rows, columns=EMOTIONS).fillna(0.0)
    totals = df.sum(axis=1)
    df["dominant_emotion"] = df[EMOTIONS].idxmax(axis=1).where(totals > 0, "none")
    return df
