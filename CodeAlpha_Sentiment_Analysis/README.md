# Task 4: Sentiment Analysis (Advanced)

Classifies text as **positive / neutral / negative**, detects **specific emotions**,
works on **Amazon-style reviews, social media posts and news**, finds **trends and
patterns in public opinion**, and turns the results into **marketing, product and
social insights**.

## How the project meets each task requirement

| Task requirement | Where it is done |
|---|---|
| 1. Classify text as positive, negative or neutral | `src/lexicon_models.py` (VADER, TextBlob), `src/ml_models.py` (TF-IDF + Logistic Regression), optional RoBERTa in `src/transformer_model.py`; combined into one **ensemble** label in `main.py` |
| 2. NLP techniques and lexicons to detect specific emotions | `src/emotions.py`: 8 emotions (joy, trust, anticipation, surprise, sadness, fear, anger, disgust) with negation handling; optional **NRC Emotion Lexicon** (`--nrc-lexicon`) or emotion transformer |
| 3. Data from Amazon reviews, social media, news | `src/data_loader.py`: CSV/Excel loader (Amazon reviews, tweets), news RSS collector (`--news`), built-in demo dataset |
| 4. Public opinion and trends via sentiment patterns | `src/trends.py` (sentiment over time, anomaly detection, root-cause aspect), source comparison, topic modelling (`src/topics.py`) |
| 5. Inform marketing, product development, social insights | `src/insights.py` produces a findings -> recommended action table in `outputs/report.md` |

## Advanced features
- **Four model families compared** on the same data (VADER, TextBlob, TF-IDF+ML, optional transformer) with accuracy, precision, recall, macro-F1 and a confusion matrix.
- **No data leakage:** ML scores for labeled rows come from cross-validation; classifiers are benchmarked (LogReg, LinearSVC, ComplementNB).
- **Ensemble** of all available models.
- **Aspect-based sentiment:** shows *what* people like/dislike (quality, price, delivery, service, design, performance, packaging, usability). Handles mixed reviews like "great design, but delivery was awful". Custom aspects via `--aspects-file`.
- **Anomaly detection** on the sentiment trend with the aspect that caused it.
- **Complaint topics** (LDA) and the most influential words per sentiment class.
- **Checkpoints / resume** after an interruption (see below).
- Mixed sources: rows without ratings (news, tweets) are still scored.

## Setup
```bash
pip install -r requirements.txt
```
(Optional deep learning: `pip install transformers torch`.)

## Run
```bash
# 1. Quick start with built-in demo data (no downloads needed)
python main.py --demo

# 2. Your own Amazon reviews (CSV or Excel)
python main.py --input reviews.csv --text-col reviewText --rating-col overall --date-col reviewTime --source-name amazon

# 3. Social media export
python main.py --input tweets.csv --text-col text --date-col created_at --source-name twitter

# 4. Add live news headlines (optionally filtered)
python main.py --demo --news --news-keyword apple

# 5. Add the transformer model / NRC lexicon
python main.py --input reviews.csv --text-col review --rating-col stars --transformer --nrc-lexicon NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

# 6. Classify one sentence (run after a full run so the ML model exists)
python main.py --predict "Great design but the delivery was awful"
```
Useful options: `--sample 5000` (use fewer rows), `--freq W` (weekly trend), `--output-dir`, `--fresh` (ignore checkpoints).

Free review datasets: Amazon Reviews on Kaggle / Hugging Face (`McAuley-Lab/Amazon-Reviews-2023`), Sentiment140 (tweets), or
the **Books to Scrape** data from Task 1 (`--input books_dataset.csv --text-col description --rating-col rating`;
note those ratings are random, so accuracy figures would not be meaningful).

## Resume from where you stopped (checkpoints)
Long runs (big review files, transformer models) can be interrupted - Ctrl+C, a crash, a closed laptop - and
**continued later by running the same command again**.

```
Progress checklist:
  [x] Collect & clean data
  [x] Lexicon models (VADER, TextBlob)
  [x] Machine-learning model (cross-validated)
  [ ] Emotion detection          <- the run stopped here; next run continues from this step
  [ ] Aspect-based sentiment
```
- Each slow step saves its result to `outputs/checkpoints/` and is ticked `[x]`; ticked steps are loaded, not recomputed.
- The transformer step also resumes **chunk by chunk** (every 500 texts), so a 2-hour job loses at most one chunk.
- If you change the input file or any option that affects results (`--sample`, `--text-col`, `--transformer`, ...),
  old checkpoints are discarded automatically, so results never get mixed.
- Add `--fresh` to ignore checkpoints and start from zero. Charts, report and Excel export are always regenerated
  (they take seconds), so changing only `--freq` or `--output-dir` still works.

## Output (`outputs/`)
| File | Content |
|---|---|
| `report.md` | Key findings + recommended actions, tables and charts |
| `results.csv` | Every text with all scores, labels, emotions |
| `summary.xlsx` | Tables: overview, model evaluation, aspects, emotions, topics, top terms, trend |
| `figures/` | 7 PNG charts (distribution, model comparison, confusion matrix, emotions, aspects, trend, top terms) |
| `models/sentiment_model.joblib` | Trained ML model |

## Project structure
```
main.py            pipeline + command line
src/               checkpoint, data_loader, preprocess, lexicon_models, ml_models, transformer_model,
                   emotions, aspects, topics, trends, visualize, insights
tests/             unit tests (python -m pytest tests)
outputs/           sample results from the demo run
```

## Important notes
- **The demo data is synthetic and template-based**, so the ML model scores ~95%.
  Expect lower numbers (roughly 75-90%) on real reviews; the demo exists to show the pipeline and the
  planted "August delivery crisis" being detected.
- Unlabeled sources (news, tweets) have no ground truth, so they are scored but excluded from accuracy metrics.
- Lexicon methods struggle with sarcasm and domain slang; the ensemble and transformer reduce, but do not remove, this.
- Check a site's terms of service before collecting real social-media or review data.
