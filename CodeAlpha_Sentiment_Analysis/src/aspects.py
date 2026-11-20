# Aspect-based sentiment analysis (ABSA).

# Overall sentiment hides detail: "Great design, but delivery was awful" is mixed.
# Here each text is split into clauses (at punctuation and at words like 'but' /
# 'however'); a clause is assigned to every aspect whose keywords it contains, and
# scored on its own with VADER. This shows WHICH parts of the product people like or
# dislike.
# 
import json
import re

import pandas as pd

from .lexicon_models import vader_scores

DEFAULT_ASPECTS = {
    "quality": "quality build material materials durable durability sturdy flimsy construction solid broke broken",
    "price": "price priced cost costs expensive affordable value money overpriced worth deal bargain",
    "delivery": "delivery delivered shipping shipped arrived arrive late delay delayed courier",
    "service": "service support customer staff refund return replacement helpful team",
    "design": "design look looks style color colour sleek beautiful size appearance ugly",
    "performance": "performance works working work fast slow battery speed lag laggy crashing stopped",
    "packaging": "packaging packaged package box wrapped",
    "usability": "easy setup install installation instructions interface intuitive simple confusing manual",
}

CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s+(?:but|however|although|though|yet)\s+", re.I)
WORD_RE = re.compile(r"[a-z']+")


def load_aspects(path=None):
    # Use built-in aspects or a JSON file like {"battery": ["battery", "charge"], ...}.# 
    if path:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return {k: set(v) for k, v in raw.items()}
    return {k: set(v.split()) for k, v in DEFAULT_ASPECTS.items()}


def extract_aspect_sentiment(df, aspects=None, text_col="text_clean"):
    # Long table: one row per (text, aspect) mention with its own sentiment score.# 
    aspects = aspects or load_aspects()
    records = []
    for idx, text, source, when in zip(df.index, df[text_col], df["source"], df["date"]):
        for clause in filter(None, (c.strip() for c in CLAUSE_SPLIT.split(text))):
            words = set(WORD_RE.findall(clause.lower()))
            for aspect, keywords in aspects.items():
                if words & keywords:
                    records.append((idx, aspect, clause, source, when))
    out = pd.DataFrame(records, columns=["row_id", "aspect", "clause", "source", "date"])
    if not out.empty:
        out["score"] = vader_scores(out["clause"])
    return out


def summarize_aspects(aspect_df, min_mentions=5):
    if aspect_df.empty:
        return pd.DataFrame()
    g = aspect_df.groupby("aspect")["score"]
    summary = pd.DataFrame({
        "mentions": g.size(),
        "mean_sentiment": g.mean(),
        "pct_positive": g.apply(lambda s: (s >= 0.05).mean() * 100),
        "pct_negative": g.apply(lambda s: (s <= -0.05).mean() * 100),
    })
    summary = summary[summary["mentions"] >= min_mentions]
    return summary.sort_values("mean_sentiment").reset_index()
