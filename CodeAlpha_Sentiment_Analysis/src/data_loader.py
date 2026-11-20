# Data collection: CSV/Excel files (e.g. Amazon reviews), news RSS feeds, and a
# built-in synthetic demo dataset so the project runs out of the box.

# Every loader returns a DataFrame with the same columns:
#     text    - the review / tweet / headline
#     rating  - numeric star rating (NaN if unavailable)
#     label   - ground-truth sentiment (positive/neutral/negative, NaN if unknown)
#     date    - datetime (NaT if unavailable)
#     source  - where the text came from (amazon, twitter, news, ...)
# 
import random
import re
from datetime import date

import numpy as np
import pandas as pd

STANDARD_COLUMNS = ["text", "rating", "label", "date", "source"]



# Labels

def rating_to_label(rating, binary=False):
    # 1-2 stars -> negative, 3 -> neutral, 4-5 -> positive.# 
    if pd.isna(rating):
        return np.nan
    if binary:
        return "positive" if rating >= 0.5 else "negative"
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def _normalize_label(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    mapping = {"pos": "positive", "positive": "positive", "1": "positive", "+1": "positive",
               "neg": "negative", "negative": "negative", "-1": "negative", "0": "neutral",
               "neu": "neutral", "neutral": "neutral"}
    return mapping.get(v, np.nan)



# File loader (Amazon reviews, tweets, any CSV/Excel with a text column)

def load_file(path, text_col, rating_col=None, date_col=None, source_col=None,
              label_col=None, source_name="file", nrows=None):
    if str(path).lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, nrows=nrows)
    else:
        raw = pd.read_csv(path, nrows=nrows, encoding_errors="replace")

    if text_col not in raw.columns:
        raise ValueError(f"Text column '{text_col}' not found. Available columns: {list(raw.columns)}")

    df = pd.DataFrame({"text": raw[text_col].astype(str)})

    df["rating"] = pd.to_numeric(raw[rating_col], errors="coerce") if rating_col else np.nan
    df["date"] = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT
    df["source"] = raw[source_col].astype(str) if source_col else source_name

    if label_col:
        df["label"] = raw[label_col].map(_normalize_label)
    elif rating_col:
        binary = df["rating"].max() <= 1
        df["label"] = df["rating"].map(lambda r: rating_to_label(r, binary))
    else:
        df["label"] = np.nan

    return df[STANDARD_COLUMNS]



# News loader (RSS) - no API key required

DEFAULT_NEWS_FEEDS = [
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
]


def fetch_news(feeds=None, keyword=None, max_items=300):
    # Collect headlines + summaries from RSS feeds (optionally filter by keyword).# 
    try:
        import feedparser
    except ImportError as exc:
        raise ImportError("Install feedparser to use --news:  pip install feedparser") from exc

    rows = []
    for url in feeds or DEFAULT_NEWS_FEEDS:
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:  # network problems should not kill the run
            print(f"  Could not read feed {url}: {exc}")
            continue
        for entry in parsed.entries:
            text = f"{entry.get('title', '')}. {re.sub('<[^>]+>', ' ', entry.get('summary', ''))}".strip()
            if keyword and keyword.lower() not in text.lower():
                continue
            published = entry.get("published_parsed")
            when = pd.Timestamp(*published[:6]) if published else pd.NaT
            rows.append({"text": text, "rating": np.nan, "label": np.nan, "date": when, "source": "news"})
    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS).drop_duplicates("text").head(max_items)
    print(f"  Collected {len(df)} news items")
    return df



# Synthetic demo data (reviews + tweets + news headlines over 12 months)

PHRASES = {
    "quality": {
        "pos": ["The build quality is excellent and feels very durable",
                "Great materials, I love how solid it feels",
                "Quality is amazing, totally worth keeping"],
        "neg": ["The quality is terrible and it broke after two days",
                "Feels cheap and flimsy, I am really disappointed",
                "Poor build quality, honestly disgusting"],
        "neu": ["The quality is about what you would expect", "It is made of plastic and metal"]},
    "price": {
        "pos": ["The price is fantastic, great value for money",
                "Very affordable and a real bargain",
                "Totally worth the price, I am delighted"],
        "neg": ["Way too expensive for what you get",
                "Overpriced and not worth the money, I feel cheated",
                "The price is ridiculous, I am so angry"],
        "neu": ["The price is similar to other brands", "It costs about thirty dollars"]},
    "delivery": {
        "pos": ["Delivery was fast and the courier was friendly",
                "It arrived early, wonderful shipping",
                "Shipping was quick and everything arrived on time"],
        "neg": ["Delivery was awful, it arrived two weeks late",
                "The shipping was delayed again and nobody told me anything",
                "Courier left it outside and it arrived damaged, terrible delivery"],
        "neu": ["It was delivered on Tuesday", "Shipping took about five days"]},
    "service": {
        "pos": ["Customer service was helpful and solved my problem quickly",
                "Support staff were friendly, I highly recommend them",
                "The refund process was easy and the team was brilliant"],
        "neg": ["Customer service was useless and never replied",
                "Support was rude and refused a refund, I am furious",
                "I waited hours for support and nobody helped"],
        "neu": ["I contacted customer service by email", "The support team replied on Monday"]},
    "design": {
        "pos": ["The design is sleek and beautiful",
                "Love the style and the color options",
                "Looks gorgeous and the size is perfect"],
        "neg": ["The design is ugly and the size is wrong",
                "Looks nothing like the photos, so disappointing",
                "Clumsy design and the color is dull"],
        "neu": ["It comes in black and white", "The design is fairly standard"]},
    "performance": {
        "pos": ["Performance is superb and it works perfectly",
                "Very fast and the battery lasts all day",
                "Runs smoothly, impressive speed"],
        "neg": ["It stopped working after a week and the battery is awful",
                "Slow and laggy, performance is terrible",
                "Does not work properly, keeps crashing and I am frustrated"],
        "neu": ["The battery lasts around eight hours", "It works as described"]},
    "packaging": {
        "pos": ["Packaging was neat and secure",
                "Beautifully packaged and nothing was damaged",
                "The box was well wrapped"],
        "neg": ["The packaging was torn and the box was crushed",
                "Poorly packaged, everything was damaged",
                "Terrible packaging, items were loose in the box"],
        "neu": ["It came in a standard cardboard box", "The package included a manual"]},
    "usability": {
        "pos": ["Setup was easy and the instructions are clear",
                "Very intuitive and simple to use",
                "Installation took five minutes, so easy"],
        "neg": ["Setup was confusing and the instructions are useless",
                "The interface is unintuitive and hard to use",
                "Impossible to install, I am so annoyed"],
        "neu": ["It has a touchscreen interface", "The instructions are in English and French"]},
}

NEWS = {
    "pos": ["Company shares rise after strong quarterly sales of its flagship product.",
            "Analysts praise the new product line for impressive performance and reliable quality.",
            "Customers welcome the lower prices and faster delivery announced this week."],
    "neg": ["Company faces backlash as customers report widespread delivery delays.",
            "Regulators raise concerns over product safety after several complaints.",
            "Shares fall as disappointing sales and poor customer service hurt the brand."],
    "neu": ["The company announced it will release its new product next month.",
            "The firm said it will hold its annual meeting on Thursday.",
            "Analysts expect the company to report quarterly results later this month."],
}

LABEL_KEY = {"positive": "pos", "negative": "neg", "neutral": "neu"}
SOURCE_PROBS = {  # (positive, neutral, negative)
    "amazon": (0.55, 0.15, 0.30),
    "twitter": (0.35, 0.25, 0.40),
    "news": (0.25, 0.50, 0.25),
}
SPIKE_MONTH = 8  # a delivery crisis in August makes a good anomaly to detect


def generate_demo_data(n=1200, seed=42):
    rnd = random.Random(seed)
    aspects = list(PHRASES)
    rows = []
    for _ in range(n):
        source = rnd.choices(["amazon", "twitter", "news"], weights=[55, 30, 15])[0]
        month = rnd.randint(1, 12)
        when = pd.Timestamp(date(2025, month, rnd.randint(1, 28)))
        probs = list(SOURCE_PROBS[source])
        crisis = month == SPIKE_MONTH
        if crisis:
            probs = [0.2, 0.15, 0.65]
        label = rnd.choices(["positive", "neutral", "negative"], weights=probs)[0]
        key = LABEL_KEY[label]

        if source == "news":
            text = rnd.choice(NEWS[key])
            if crisis and label == "negative" and rnd.random() < 0.7:
                text = "Company faces backlash as customers report widespread delivery delays."
            rows.append((text, np.nan, label, when, source))
            continue

        sentences = []
        first_aspect = "delivery" if (crisis and label == "negative" and rnd.random() < 0.7) else rnd.choice(aspects)
        sentences.append(rnd.choice(PHRASES[first_aspect][key]))
        for _ in range(rnd.choice([0, 1, 1, 2])):
            other = rnd.choice(aspects)
            # some reviews are mixed: praise one thing, criticise another
            k = key
            if label != "neutral" and rnd.random() < 0.25:
                k = "neg" if label == "positive" else "pos"
            sentences.append(rnd.choice(PHRASES[other][k]))
        text = ". ".join(sentences)
        text = text + ("." if rnd.random() < 0.8 else "!")

        rating = np.nan
        if source == "amazon":
            rating = {"pos": rnd.choice([4, 5, 5]), "neu": 3, "neg": rnd.choice([1, 1, 2])}[key]
        else:  # tweet style
            text = rnd.choice(["", "lol ", "omg ", "ugh ", "Honestly "]) + text
            text += {"pos": " #love", "neg": " #fail", "neu": " #review"}[key] + " @BrandName https://t.co/xyz"
        rows.append((text, rating, label, when, source))

    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)
