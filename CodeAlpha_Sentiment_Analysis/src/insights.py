# Turn the numbers into plain-language findings and a Markdown report. 
from datetime import datetime


def _md_table(df, floatfmt="{:.2f}"):
    if df is None or len(df) == 0:
        return "_No data._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def net_sentiment(series):
    # Net Sentiment Score = % positive - % negative.# 
    return ((series == "positive").mean() - (series == "negative").mean()) * 100


def build_insights(df, aspect_summary, emotion_by_sentiment, anomalies, eval_df):
    out = []
    shares = df["sentiment"].value_counts(normalize=True) * 100
    nss = net_sentiment(df["sentiment"])
    out.append(("Overview", f"{shares.get('positive', 0):.0f}% positive, {shares.get('neutral', 0):.0f}% neutral, "
                            f"{shares.get('negative', 0):.0f}% negative (Net Sentiment Score {nss:+.0f}).",
                "Track the Net Sentiment Score monthly as a headline KPI."))

    by_source = df.groupby("source")["sentiment"].apply(net_sentiment).sort_values()
    if len(by_source) > 1:
        out.append(("Social insights", f"'{by_source.index[0]}' is the most negative channel "
                    f"(NSS {by_source.iloc[0]:+.0f}); '{by_source.index[-1]}' is the most positive "
                    f"(NSS {by_source.iloc[-1]:+.0f}).",
                    f"Investigate why opinion differs on '{by_source.index[0]}' and tailor messaging per channel."))

    if aspect_summary is not None and len(aspect_summary) >= 2:
        worst, best = aspect_summary.iloc[0], aspect_summary.iloc[-1]
        out.append(("Product development",
                    f"'{worst['aspect']}' is the weakest aspect (mean {worst['mean_sentiment']:+.2f}, "
                    f"{worst['pct_negative']:.0f}% negative over {int(worst['mentions'])} mentions).",
                    f"Prioritise fixing '{worst['aspect']}' in the next release / operations review."))
        out.append(("Marketing",
                    f"'{best['aspect']}' is the strongest aspect (mean {best['mean_sentiment']:+.2f}, "
                    f"{best['pct_positive']:.0f}% positive).",
                    f"Lead campaigns and ad copy with '{best['aspect']}'; reuse customer wording as testimonials."))

    if emotion_by_sentiment is not None and "negative" in emotion_by_sentiment.index:
        neg_em = emotion_by_sentiment.loc["negative"].sort_values(ascending=False)
        if neg_em.iloc[0] > 0:
            out.append(("Customer support", f"Negative texts are dominated by '{neg_em.index[0]}' "
                        f"(then '{neg_em.index[1]}').",
                        "Train support replies to acknowledge this emotion first (apology, empathy, concrete fix)."))
    if emotion_by_sentiment is not None and "positive" in emotion_by_sentiment.index:
        pos_em = emotion_by_sentiment.loc["positive"].sort_values(ascending=False)
        if pos_em.iloc[0] > 0:
            out.append(("Marketing", f"Positive texts are dominated by '{pos_em.index[0]}'.",
                        f"Use a '{pos_em.index[0]}'-focused tone in brand messaging."))

    for a in anomalies or []:
        when = a["period"].strftime("%B %Y")
        driver = f" Main driver: '{a['driver']}' (mean {a['driver_score']:+.2f})." if a["driver"] else ""
        if a["direction"] == "dip":
            out.append(("Risk alert", f"Sentiment dipped sharply in {when} (z = {a['z_score']:+.1f}).{driver}",
                        "Review what changed in that period (logistics, release, campaign) and set up an alert."))
        else:
            out.append(("Opportunity", f"Sentiment peaked in {when} (z = {a['z_score']:+.1f}).{driver}",
                        "Find out what worked (promotion, launch) and repeat it."))

    if eval_df is not None and len(eval_df):
        best_model = eval_df.sort_values("f1_macro", ascending=False).iloc[0]
        out.append(("Model quality", f"Best model on labeled data: {best_model['model']} "
                    f"(accuracy {best_model['accuracy']:.2f}, macro-F1 {best_model['f1_macro']:.2f}).",
                    "Use the ensemble/ML model for reporting; spot-check low-confidence rows manually."))
    return out


def write_report(path, df, insights, tables, figure_names):
    lines = ["# Sentiment Analysis Report", "",
             f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Texts analysed: **{len(df)}**._", "",
             "## Key findings and recommended actions", "",
             "| Area | Finding | Recommended action |", "|---|---|---|"]
    for area, finding, action in insights:
        lines.append(f"| {area} | {finding} | {action} |")
    for title, table in tables:
        lines += ["", f"## {title}", "", _md_table(table)]
    lines += ["", "## Charts", ""]
    lines += [f"![{n}](figures/{n})" for n in figure_names]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
