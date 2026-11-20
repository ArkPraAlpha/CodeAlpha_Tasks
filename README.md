# 📊 CodeAlpha Data Analytics Internship Projects

> Projects completed during my **Data Analytics Internship at CodeAlpha** — covering web scraping and advanced sentiment analysis with machine learning.

---

## 👨‍💻 Intern Information

| Field | Details |
|-------|---------|
| **Name** | Ark Prajapati |
| **Organization** | CodeAlpha |
| **Domain** | Data Analytics |
| **Duration** | 20 October 2026 – 20 November 2026 |

---

## 📌 Completed Tasks

### ✅ Task 1 — Web Scraping ([`CodeAlpha_Web_Scraping/`](./CodeAlpha_Web_Scraping))

A Python scraper that automatically collects data for all **1,000 books** on [Books to Scrape](https://books.toscrape.com/).

- Pagination handled automatically (50 catalogue pages + 1,000 book detail pages)
- Extracts title, price, rating, category, description, stock, UPC, and image URL
- Automatic retries (up to 3×) with polite delays between requests
- Output saved as both **CSV** and **Excel**

### ✅ Task 4 — Sentiment Analysis ([`CodeAlpha_Sentiment_Analysis/`](./CodeAlpha_Sentiment_Analysis))

An advanced multi-model sentiment analysis pipeline for Amazon reviews, tweets, and news.

- **Four model families**: VADER, TextBlob, TF-IDF + Logistic Regression, optional RoBERTa transformer
- **Ensemble** of all available models
- **Emotion detection** (joy, trust, anger, fear, sadness, disgust, surprise, anticipation)
- **Aspect-based sentiment**: detects *what* customers like/dislike (quality, delivery, service, etc.)
- **Trend analysis** with anomaly detection
- **LDA topic modelling** on negative reviews
- **Checkpoint / resume** support for large datasets
- Outputs: `report.md`, `results.csv`, `summary.xlsx`, 7 PNG charts

---

## 🛠️ Technologies Used

| Category | Libraries |
|----------|-----------|
| **Data** | `pandas`, `numpy` |
| **ML / NLP** | `scikit-learn`, `vaderSentiment`, `textblob` |
| **Deep Learning** | `transformers`, `torch` *(optional)* |
| **Visualisation** | `matplotlib` |
| **Web Scraping** | `requests`, `beautifulsoup4` |
| **Utilities** | `joblib`, `openpyxl`, `feedparser` |

---

## 📂 Repository Structure

```
CodeAlpha/
├── CodeAlpha_Web_Scraping/
│   ├── books_scraper.py        ← main scraper script
│   ├── requirements.txt
│   └── README.md
├── CodeAlpha_Sentiment_Analysis/
│   ├── main.py                 ← pipeline entry point
│   ├── src/                    ← modular source code
│   │   ├── data_loader.py
│   │   ├── preprocess.py
│   │   ├── lexicon_models.py
│   │   ├── ml_models.py
│   │   ├── transformer_model.py
│   │   ├── emotions.py
│   │   ├── aspects.py
│   │   ├── topics.py
│   │   ├── trends.py
│   │   ├── visualize.py
│   │   ├── insights.py
│   │   └── checkpoint.py
│   ├── tests/
│   │   └── test_basic.py
│   ├── requirements.txt
│   └── README.md
├── .gitignore
└── README.md                   ← this file
```

---

## 🚀 Quick Start

### Task 1 — Web Scraping
```bash
cd CodeAlpha_Web_Scraping
pip install -r requirements.txt
python books_scraper.py
```

### Task 4 — Sentiment Analysis
```bash
cd CodeAlpha_Sentiment_Analysis
pip install -r requirements.txt

# Run with built-in demo data (no downloads needed)
python main.py --demo

# Run with the scraped books dataset
python main.py --input ../CodeAlpha_Web_Scraping/books_dataset.csv \
               --text-col description --rating-col rating

# Classify a single sentence
python main.py --predict "Great quality but the delivery was awful"
```

---

## 🎯 Learning Outcomes

- Real-world web scraping with pagination, retries, and polite crawling
- NLP pipeline design: preprocessing → lexicon → ML → transformer → ensemble
- Aspect-based sentiment and emotion detection
- Anomaly detection on time-series sentiment data
- Modular, testable Python code with checkpoint / resume support
- Data export to CSV, Excel, and Markdown reports

---

## 🙏 Acknowledgement

I sincerely thank **CodeAlpha** for providing this internship opportunity and helping me build practical skills in Data Analytics and Machine Learning.

---

## 👨‍💻 Author

**Ark Prajapati**
GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

*#CodeAlpha #DataAnalytics #Python #NLP #SentimentAnalysis #WebScraping #MachineLearning #Internship*