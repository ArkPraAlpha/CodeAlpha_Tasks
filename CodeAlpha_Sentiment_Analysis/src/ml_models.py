"""Supervised machine learning: TF-IDF + classifier.

* Several classifiers are benchmarked with cross-validation.
* Logistic Regression is used to produce a sentiment score for every row. For rows
  that have a ground-truth label the score comes from cross_val_predict (so the model
  never scores a text it was trained on - no data leakage); unlabeled rows (e.g. news,
  tweets without stars) are scored by a model trained on all labeled rows.
* The most influential words per class are exported for interpretation.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def _pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def benchmark_classifiers(texts, labels, n_splits=5, seed=42):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "ComplementNB": ComplementNB(),
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, clf in models.items():
        res = cross_validate(_pipeline(clf), texts, labels, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append({"classifier": name,
                     "accuracy_mean": res["test_accuracy"].mean(),
                     "accuracy_std": res["test_accuracy"].std(),
                     "f1_macro_mean": res["test_f1_macro"].mean(),
                     "f1_macro_std": res["test_f1_macro"].std()})
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def _signed_score(proba, classes):
    classes = list(classes)
    pos = proba[:, classes.index("positive")] if "positive" in classes else 0.0
    neg = proba[:, classes.index("negative")] if "negative" in classes else 0.0
    return pos - neg


def top_features(model, n=15):
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = np.array(tfidf.get_feature_names_out())
    classes = list(clf.classes_)
    coef = clf.coef_
    if len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    rows = []
    for i, cls in enumerate(classes):
        for word, weight in zip(names[np.argsort(coef[i])[::-1][:n]], np.sort(coef[i])[::-1][:n]):
            rows.append({"class": cls, "term": word, "weight": float(weight)})
    return pd.DataFrame(rows)


def fit_ml_scores(df, text_col="text_ml", label_col="label", seed=42, n_splits=5):
    """Return dict(score, pred, benchmark, features, model) or None if there is too little labeled data."""
    labeled = df[label_col].notna()
    if labeled.sum() < 30 or df.loc[labeled, label_col].nunique() < 2:
        print("  Not enough labeled data for supervised model - skipping ML.")
        return None

    X = df.loc[labeled, text_col].values
    y = df.loc[labeled, label_col].values
    n_splits = int(min(n_splits, pd.Series(y).value_counts().min()))
    if n_splits < 2:
        return None

    bench = benchmark_classifiers(X, y, n_splits, seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = _pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"))

    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = sorted(set(y))
    score = pd.Series(np.nan, index=df.index)
    pred = pd.Series(pd.NA, index=df.index, dtype="object")
    score[labeled] = _signed_score(oof, classes)
    pred[labeled] = np.array(classes)[oof.argmax(axis=1)]

    model.fit(X, y)
    if (~labeled).any():
        proba = model.predict_proba(df.loc[~labeled, text_col].values)
        score[~labeled] = _signed_score(proba, model.classes_)
        pred[~labeled] = model.classes_[proba.argmax(axis=1)]

    return {"score": score, "pred": pred, "benchmark": bench,
            "features": top_features(model), "model": model}
