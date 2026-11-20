import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.aspects import extract_aspect_sentiment, summarize_aspects
from src.data_loader import generate_demo_data, rating_to_label
from src.emotions import emotion_counts
from src.lexicon_models import score_to_label, vader_scores
from src.preprocess import add_text_columns, clean_text


def test_clean_text_removes_noise():
    assert clean_text("Love it @bob https://x.co/1 #great <b>buy</b>") == "Love it great buy"


def test_rating_to_label():
    assert [rating_to_label(r) for r in (1, 3, 5)] == ["negative", "neutral", "positive"]


def test_vader_direction():
    s = vader_scores(["I love this, it is excellent!", "This is terrible and I hate it."])
    assert s[0] > 0.05 and s[1] < -0.05
    assert list(score_to_label(s)) == ["positive", "negative"]


def test_emotions_and_negation():
    assert emotion_counts("I am so angry and furious")["anger"] == 2
    assert emotion_counts("I am not happy")["sadness"] == 1  # negated joy


def test_aspect_splits_mixed_review():
    df = add_text_columns(generate_demo_data(10)).iloc[:1].copy()
    df["text_clean"] = "The design is gorgeous, but delivery was awful."
    df["date"] = None
    out = extract_aspect_sentiment(df)
    scores = dict(zip(out["aspect"], out["score"]))
    assert scores["design"] > 0 > scores["delivery"]


def test_demo_data_shape():
    d = generate_demo_data(100)
    assert len(d) == 100 and set(d["label"]) <= {"positive", "neutral", "negative"}


def test_checkpoint_resume_and_invalidation(tmp_path):
    from src.checkpoint import Checkpoint
    calls = []
    ck = Checkpoint(tmp_path, {"a": 1})
    assert ck.run("step", "Step", lambda: calls.append(1) or 42) == 42
    ck2 = Checkpoint(tmp_path, {"a": 1})              # same settings -> loaded, not recomputed
    assert ck2.run("step", "Step", lambda: calls.append(1) or 0) == 42 and len(calls) == 1
    ck3 = Checkpoint(tmp_path, {"a": 2})              # changed settings -> discarded
    assert not ck3.done("step")
