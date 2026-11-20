# Task 1: Web Scraping - Books to Scrape

A Python scraper that automatically collects data for all 1,000 books on
[Books to Scrape](https://books.toscrape.com/), a practice website built for
learning web scraping. It walks through every catalogue page, opens each book's
detail page through its link, extracts the information, and saves one row per
book to CSV and Excel.

## Project Files

| File | Purpose |
|------|---------|
| `books_scraper.py` | The scraper (main script) |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |
| `books_dataset.csv` | Output dataset (created when you run the script) |
| `books_dataset.xlsx` | Same dataset in Excel format (created when you run the script) |
| `failed_urls.csv` | Only created if some pages fail to download |

## Setup

Requires Python 3.8 or newer.

```bash
# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python books_scraper.py
```

The run takes a few minutes because it makes about 1,050 requests (50 catalogue
pages + 1,000 book pages) with a short polite delay between them. Progress is
printed as it goes, and a summary is shown at the end.

## How It Works

1. **Open the catalogue page.** The script starts at page 1 and finds the 20
   book listings on it.
2. **Collect each book's link.** It reads the `href` of every listing from the
   HTML and converts it to a full URL, so no manual clicking is needed. It then
   follows the "next" button until the last page.
3. **Visit every book detail page.** It extracts the description and the
   product information table.
4. **Store everything.** All records are put in a pandas DataFrame and saved as
   CSV and Excel.

## Dataset Columns

| Column | Description |
|--------|-------------|
| `title` | Book title |
| `price` | Displayed price (GBP, numeric) |
| `rating` | Star rating converted to a number from 1 to 5 |
| `availability` | Stock status text, e.g. "In stock (22 available)" |
| `stock_quantity` | Number of copies available (numeric) |
| `description` | Full product description (may be empty for some books) |
| `category` | Genre, e.g. Poetry |
| `upc` | Unique product code |
| `product_type` | Product type |
| `price_excl_tax` | Price excluding tax |
| `price_incl_tax` | Price including tax |
| `tax` | Tax amount |
| `review_count` | Number of reviews |
| `image_url` | Link to the cover image |
| `product_url` | Link to the book's detail page |

### Sample Output

| title | price | rating | category | review_count |
|-------|-------|--------|----------|--------------|
| A Light in the Attic | 51.77 | 3 | Poetry | 0 |
| Tipping the Velvet | 53.74 | 1 | Historical Fiction | 0 |
| Soumission | 50.10 | 1 | Fiction | 0 |

## Features

- Pagination handled automatically (all 50 pages).
- Retries failed requests up to 3 times.
- A single failed page does not stop the run; failures are logged to
  `failed_urls.csv`.
- UTF-8 handling so prices show as `51.77` rather than `Â£51.77`.
- Numeric columns (price, tax, rating, reviews) are cleaned and ready for
  analysis.
- Summary report at the end of each run.

## Configuration

Edit the constants at the top of `books_scraper.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `DELAY_SECONDS` | `0.2` | Pause between requests |
| `MAX_RETRIES` | `3` | Attempts per failed request |
| `OUTPUT_CSV` | `books_dataset.csv` | CSV file name |
| `OUTPUT_XLSX` | `books_dataset.xlsx` | Excel file name |

## Limitations

- **No author field.** The site does not provide authors, so there is no
  author column.
- **No review text.** Only the number of reviews is available, not individual
  reviews.
- **Demo data.** The website states that prices and ratings are randomly
  assigned for demonstration, so they do not reflect real market data.

## Ethical Scraping Note

This site is intended for scraping practice. The script identifies itself with a
User-Agent and pauses between requests. For any real website, check its
`robots.txt` and terms of service before scraping.

## Expected Result

After a successful run, the summary should show **1,000 books saved** and
**0 failed pages**.
