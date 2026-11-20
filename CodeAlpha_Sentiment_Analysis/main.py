"""
Advanced Sentiment Analysis pipeline.

Examples
  python main.py --demo
  python main.py --input amazon_reviews.csv --text-col reviewText --rating-col overall --date-col reviewTime
  python main.py --input tweets.csv --text-col text --source-name twitter --date-col created_at
  python main.py --demo --news --news-keyword apple
  python main.py --input reviews.csv --text-col review --rating-col stars --transformer
  python main.py --predict "Great phone but the battery is awful"
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src import data_loader, emotions as emo, insights, visualize
from src.checkpoint import Checkpoint
from src.aspects import extract_aspect_sentiment, load_aspects, summarize_aspects
from src.lexicon_models import score_to_label, textblob_scores, vader_scores
from src.ml_models import fit_ml_scores
from src.preprocess import add_text_columns, clean_text
from src.topics import lda_topics
from src.trends import detect_anomalies, explain_anomalies, sentiment_over_time

WEIGHTS = {"vader": 1.0, "textblob": 0.5, "ml": 1.5, "transformer": 2.0}
ENSEMBLE_THRESHOLD = 0.10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="use the built-in synthetic dataset")
    p.add_argument("--demo-size", type=int, default=1200)
    p.add_argument("--input", help="CSV/Excel file with reviews, tweets, etc.")
    p.add_argument("--text-col", default="text")
    p.add_argument("--rating-col")
    p.add_argument("--label-col", help="column with positive/neutral/negative ground truth")
    p.add_argument("--date-col")
    p.add_argument("--source-col")
    p.add_argument("--source-name", default="file")
    p.add_argument("--sample", type=int, help="randomly use only N rows (faster)")
    p.add_argument("--news", action="store_true", help="also collect headlines from news RSS feeds")
    p.add_argument("--news-keyword", help="keep only news items containing this word")
    p.add_argument("--transformer", action="store_true", help="add a RoBERTa sentiment model")
    p.add_argument("--transformer-model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    p.add_argument("--emotion-transformer", action="store_true", help="use a transformer for emotions")
    p.add_argument("--nrc-lexicon", help="path to NRC word-level emotion lexicon file")
    p.add_argument("--aspects-file", help="JSON file with custom aspects")
    p.add_argument("--freq", default="M", choices=["M", "W", "Q"], help="trend period")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--fresh", action="store_true",
                   help="ignore saved checkpoints and start from scratch (default: resume)")
    p.add_argument("--predict", help="classify a single text and exit")
    return p.parse_args()


# ----------------------------------------------------------------------------
def predict_single(text, out_dir):
    text = clean_text(text)
    v = float(vader_scores([text])[0])
    print(f"\nText   : {text}")
    print(f"VADER  : {v:+.3f} -> {score_to_label([v])[0]}")
    t = textblob_scores([text])
    if t is not None:
        print(f"TextBlob: {t[0]:+.3f} -> {score_to_label(t)[0]}")
    model_path = Path(out_dir) / "models" / "sentiment_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        probs = model.predict_proba([text.lower()])[0]
        print("ML     : " + ", ".join(f"{c}={p:.2f}" for c, p in zip(model.classes_, probs)))
    counts = {k: v for k, v in emo.emotion_counts(text).items() if v}
    print(f"Emotions: {counts or 'none detected'}")
    for _, r in extract_aspect_sentiment(
            pd.DataFrame({"text_clean": [text], "source": ["input"], "date": [pd.NaT]})).iterrows():
        print(f"Aspect : {r['aspect']:<12} {r['score']:+.2f}  <- \"{r['clause']}\"")


def collect_data(args):
    frames = []
    if args.demo:
        frames.append(data_loader.generate_demo_data(args.demo_size))
    if args.input:
        frames.append(data_loader.load_file(args.input, args.text_col, args.rating_col, args.date_col,
                                            args.source_col, args.label_col, args.source_name))
    if args.news:
        frames.append(data_loader.fetch_news(keyword=args.news_keyword))
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("No data. Use --demo, --input FILE, or --news (see --help).")
    df = pd.concat(frames, ignore_index=True)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
    return df


def evaluate(df, pred_cols):
    labeled = df[df["label"].notna()]
    if labeled.empty:
        return pd.DataFrame()
    rows = []
    for name, col in pred_cols.items():
        sub = labeled[labeled[col].notna()]
        if sub.empty:
            continue
        rows.append({"model": name, "n": len(sub),
                     "accuracy": accuracy_score(sub["label"], sub[col]),
                     "precision": precision_score(sub["label"], sub[col], average="macro", zero_division=0),
                     "recall": recall_score(sub["label"], sub[col], average="macro", zero_division=0),
                     "f1_macro": f1_score(sub["label"], sub[col], average="macro", zero_division=0)})
    return pd.DataFrame(rows)


STEPS = [("data", "Collect & clean data"),
         ("lexicon", "Lexicon models (VADER, TextBlob)"),
         ("ml", "Machine-learning model (cross-validated)"),
         ("transformer", "Transformer sentiment model (optional)"),
         ("emotions", "Emotion detection"),
         ("aspects", "Aspect-based sentiment")]


def fingerprint(args):
    """Settings that change the results; if any differ, old checkpoints are not reused."""
    keys = ["demo", "demo_size", "input", "text_col", "rating_col", "label_col", "date_col",
            "source_col", "source_name", "sample", "news", "news_keyword", "transformer",
            "transformer_model", "emotion_transformer", "nrc_lexicon", "aspects_file"]
    fp = {k: getattr(args, k) for k in keys}
    for k in ("input", "nrc_lexicon", "aspects_file"):  # also detect an edited file
        if fp[k] and Path(fp[k]).exists():
            stat = Path(fp[k]).stat()
            fp[k + "_stat"] = [stat.st_size, int(stat.st_mtime)]
    return fp


def run(args):
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(out_dir, fingerprint(args), fresh=args.fresh)
    steps = [(k, t) for k, t in STEPS if k != "transformer" or args.transformer]
    ck.show(steps)

    # 1. data ---------------------------------------------------------------
    df = ck.run("data", "Collect & clean data", lambda: add_text_columns(collect_data(args)))
    print(f"      {len(df)} texts | sources: {df['source'].value_counts().to_dict()} "
          f"| labeled: {int(df['label'].notna().sum())}")

    # 2. lexicon models -----------------------------------------------------
    def lexicon_step():
        out = pd.DataFrame(index=df.index)
        out["score_vader"] = vader_scores(df["text_clean"])
        out["pred_vader"] = score_to_label(out["score_vader"])
        tb = textblob_scores(df["text_clean"])
        if tb is not None:
            out["score_textblob"] = tb
            out["pred_textblob"] = score_to_label(tb)
        return out
    lex = ck.run("lexicon", "Lexicon models (VADER, TextBlob)", lexicon_step)
    df = df.join(lex)

    # 3. machine learning ---------------------------------------------------
    ml = ck.run("ml", "Machine-learning model (cross-validated)", lambda: fit_ml_scores(df))
    benchmark = features = None
    if ml:
        df["score_ml"], df["pred_ml"] = ml["score"], ml["pred"]
        benchmark, features = ml["benchmark"], ml["features"]
        joblib.dump(ml["model"], out_dir / "models" / "sentiment_model.joblib")

    # 3b. transformer (slow -> also resumes part-way through, chunk by chunk)
    if args.transformer:
        def transformer_step():
            from src.transformer_model import transformer_sentiment
            sc, lb = transformer_sentiment(df["text_clean"], args.transformer_model,
                                           cache_dir=ck.chunk_dir("transformer_chunks"))
            return pd.DataFrame({"score_transformer": sc, "pred_transformer": lb}, index=df.index)
        df = df.join(ck.run("transformer", "Transformer sentiment model", transformer_step))

    # 4. ensemble (instant, always recomputed) ------------------------------
    present = [k for k in WEIGHTS if f"score_{k}" in df]
    scores = df[[f"score_{k}" for k in present]].to_numpy(dtype=float)
    w = np.array([WEIGHTS[k] for k in present])
    mask = ~np.isnan(scores)
    df["ensemble_score"] = (np.nan_to_num(scores) * w * mask).sum(axis=1) / (w * mask).sum(axis=1)
    df["sentiment"] = score_to_label(df["ensemble_score"], ENSEMBLE_THRESHOLD, -ENSEMBLE_THRESHOLD)
    print(f"      ensemble of: {present}")

    pred_cols = {"VADER": "pred_vader", "TextBlob": "pred_textblob", "ML (TF-IDF+LR)": "pred_ml",
                 "Transformer": "pred_transformer", "Ensemble": "sentiment"}
    eval_df = evaluate(df, {k: v for k, v in pred_cols.items() if v in df})

    # 5. emotions -----------------------------------------------------------
    def emotion_step():
        if args.nrc_lexicon:
            emo.load_nrc_lexicon(args.nrc_lexicon)
        if args.emotion_transformer:
            from src.transformer_model import transformer_emotions
            table = transformer_emotions(df["text_clean"])
            cols = [c for c in table.columns if c not in ("dominant_emotion", "neutral")]
        else:
            table, cols = emo.detect_emotions(df["text_clean"]), emo.EMOTIONS
        return table, cols
    em, emotion_cols = ck.run("emotions", "Emotion detection", emotion_step)
    df["dominant_emotion"] = em["dominant_emotion"].values
    for c in emotion_cols:
        df[f"emo_{c}"] = em[c].values
    emotion_by_sentiment = df.groupby("sentiment")[[f"emo_{c}" for c in emotion_cols]].mean()
    emotion_by_sentiment.columns = emotion_cols

    # 6. aspects ------------------------------------------------------------
    aspect_df = ck.run("aspects", "Aspect-based sentiment",
                       lambda: extract_aspect_sentiment(df, load_aspects(args.aspects_file)))
    aspect_summary = summarize_aspects(aspect_df)

    # 7. topics & trends (fast, always recomputed) --------------------------
    print("  Topics, trends, charts and report (fast - always regenerated) ...")
    topics = lda_topics(df.loc[df["sentiment"] == "negative", "text_clean"].tolist())
    ts = anomalies = by_source = None
    explained = []
    if df["date"].notna().sum() >= 20:
        ts = sentiment_over_time(df, freq=args.freq)
        if len(ts):
            ts = detect_anomalies(ts)
            explained = explain_anomalies(ts, aspect_df, args.freq)
            by_source = {s: sentiment_over_time(g, freq=args.freq)
                         for s, g in df.groupby("source") if g["date"].notna().sum() >= 20}

    # 8. charts, report, exports --------------------------------------------
    figs = [visualize.plot_sentiment_distribution(df, fig_dir).name]
    if len(eval_df):
        figs.append(visualize.plot_model_comparison(eval_df, fig_dir).name)
        lab = df[df["label"].notna()]
        figs.append(visualize.plot_confusion(lab["label"], lab["sentiment"], fig_dir).name)
    figs.append(visualize.plot_emotions(emotion_by_sentiment, fig_dir).name)
    if len(aspect_summary):
        figs.append(visualize.plot_aspects(aspect_summary, fig_dir).name)
    if ts is not None and len(ts):
        figs.append(visualize.plot_trend(ts, fig_dir, by_source).name)
    if features is not None:
        figs.append(visualize.plot_top_terms(features, fig_dir).name)

    found = insights.build_insights(df, aspect_summary if len(aspect_summary) else None,
                                    emotion_by_sentiment, explained, eval_df if len(eval_df) else None)

    overview = df.groupby("source")["sentiment"].agg(
        texts="size",
        pct_positive=lambda s: (s == "positive").mean() * 100,
        pct_neutral=lambda s: (s == "neutral").mean() * 100,
        pct_negative=lambda s: (s == "negative").mean() * 100,
        net_sentiment=insights.net_sentiment).reset_index()
    tables = [("Sentiment by source", overview), ("Model evaluation", eval_df),
              ("Classifier benchmark (5-fold CV)", benchmark),
              ("Aspect-based sentiment", aspect_summary),
              ("Emotions by sentiment (avg. words per text)", emotion_by_sentiment.round(3).reset_index()),
              ("Complaint topics (LDA on negative texts)", topics)]
    if ts is not None:
        tables.append(("Sentiment over time", ts.assign(period=ts["period"].dt.strftime("%Y-%m")).round(3)))
    insights.write_report(out_dir / "report.md", df, found, tables, figs)

    df.drop(columns=["text_ml"]).to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "summary.xlsx") as xl:
        for name, table in [("overview", overview), ("model_eval", eval_df), ("benchmark", benchmark),
                            ("aspects", aspect_summary), ("emotions", emotion_by_sentiment.reset_index()),
                            ("topics", topics), ("top_terms", features),
                            ("trend", ts if ts is not None else pd.DataFrame())]:
            if table is not None and len(table):
                table.to_excel(xl, sheet_name=name, index=False)

    print("\n================ SUMMARY ================")
    for area, finding, action in found:
        print(f"[{area}] {finding}\n    -> {action}")
    print(f"\nSaved to {out_dir}/: results.csv, summary.xlsx, report.md, figures/ ({len(figs)} charts)")
    ck.show(steps)
    return df


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.predict:
        predict_single(arguments.predict, arguments.output_dir)
    else:
        try:
            run(arguments)
        except KeyboardInterrupt:
            print("\n\nStopped. Finished steps are saved - run the same command again to resume (add --fresh to start over).")
