"""Optional deep-learning models from Hugging Face (needs: pip install transformers torch).

* Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest (positive/neutral/negative)
* Emotion:   j-hartmann/emotion-english-distilroberta-base (7 emotions)

The first call downloads the model (a few hundred MB) and needs internet access.
"""
import numpy as np
import pandas as pd


def _pipeline(task_model, device):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install transformers and torch to use transformer models:\n"
                          "  pip install transformers torch") from exc
    return pipeline("text-classification", model=task_model, top_k=None,
                    truncation=True, max_length=256, device=device)


def transformer_sentiment(texts, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
                          batch_size=32, device=-1, chunk_size=500, cache_dir=None):
    """Return (signed score = P(pos) - P(neg), predicted label).

    Texts are processed in chunks of `chunk_size`. If `cache_dir` is given, every
    finished chunk is saved there, so an interrupted run continues at the first
    unfinished chunk instead of starting over.
    """
    import joblib
    from pathlib import Path

    texts = list(texts)
    clf = None
    scores, labels = [], []
    for start in range(0, len(texts), chunk_size):
        chunk_file = Path(cache_dir) / f"chunk_{start}.joblib" if cache_dir else None
        if chunk_file is not None and chunk_file.exists():
            chunk_scores, chunk_labels = joblib.load(chunk_file)
        else:
            clf = clf or _pipeline(model_name, device)
            chunk_scores, chunk_labels = [], []
            for result in clf(texts[start:start + chunk_size], batch_size=batch_size):
                probs = {r["label"].lower(): r["score"] for r in result}
                chunk_scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
                chunk_labels.append(max(probs, key=probs.get))
            if chunk_file is not None:
                joblib.dump((chunk_scores, chunk_labels), chunk_file)
            print(f"      transformer: {min(start + chunk_size, len(texts))}/{len(texts)} texts done")
        scores.extend(chunk_scores)
        labels.extend(chunk_labels)
    return np.array(scores), np.array(labels)


def transformer_emotions(texts, model_name="j-hartmann/emotion-english-distilroberta-base",
                         batch_size=32, device=-1):
    """Return a DataFrame of emotion probabilities plus 'dominant_emotion'."""
    clf = _pipeline(model_name, device)
    rows = [{r["label"].lower(): r["score"] for r in res}
            for res in clf(list(texts), batch_size=batch_size)]
    df = pd.DataFrame(rows)
    df["dominant_emotion"] = df.idxmax(axis=1)
    return df
