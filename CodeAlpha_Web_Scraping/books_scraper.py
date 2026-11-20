# TASK 1: Web Scraping - Books to Scrape (https://books.toscrape.com/)

import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Configuration

BASE_URL = "https://books.toscrape.com/"
START_URL = urljoin(BASE_URL, "catalogue/page-1.html")
DELAY_SECONDS = 0.2          # polite pause between requests
MAX_RETRIES = 3
OUTPUT_CSV = "books_dataset.csv"
OUTPUT_XLSX = "books_dataset.xlsx"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

session = requests.Session()
session.headers.update({"User-Agent": "EducationalBookScraper/1.0"})



# Helpers

def get_soup(url):
    """Download a page (with retries) and return a BeautifulSoup object."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            response.encoding = "utf-8"  # prevents '£' turning into 'Â£'
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Retry {attempt}/{MAX_RETRIES} for {url}: {exc}")
            time.sleep(2 * attempt)


def get_text(element):
    return element.get_text(" ", strip=True) if element else ""


def to_float(text):
    """'£51.77' -> 51.77"""
    match = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(match.group()) if match else None


def to_int(text):
    match = re.search(r"\d+", text or "")
    return int(match.group()) if match else None



# Steps 1 & 2: walk the catalogue and collect every book URL

def collect_book_urls():
    book_urls = []
    page_url = START_URL
    page_number = 1

    while page_url:
        soup = get_soup(page_url)
        links = soup.select("article.product_pod h3 a")
        for link in links:
            book_urls.append(urljoin(page_url, link["href"]))
        print(f"Catalogue page {page_number}: {len(links)} books found")

        next_link = soup.select_one("li.next a")
        page_url = urljoin(page_url, next_link["href"]) if next_link else None
        page_number += 1
        time.sleep(DELAY_SECONDS)

    print(f"\nTotal book links collected: {len(book_urls)}\n")
    return book_urls



# Step 3: extract details from one book page

def get_book_details(book_url):
    soup = get_soup(book_url)

    # Product information table (UPC, type, prices, tax, availability, reviews)
    info = {}
    for row in soup.select("table.table tr"):
        header, value = row.find("th"), row.find("td")
        if header and value:
            info[get_text(header)] = get_text(value)

    # Rating: class looks like "star-rating Three"
    rating_tag = soup.select_one("p.star-rating")
    rating = None
    if rating_tag:
        for css_class in rating_tag.get("class", []):
            if css_class in RATING_MAP:
                rating = RATING_MAP[css_class]

    # Category: breadcrumb is Home > Books > <Category> > <Title>
    breadcrumb = soup.select("ul.breadcrumb li a")
    category = get_text(breadcrumb[2]) if len(breadcrumb) >= 3 else ""

    # Description: the <p> right after the "Product Description" header
    description = ""
    desc_header = soup.select_one("#product_description")
    if desc_header:
        description = get_text(desc_header.find_next_sibling("p"))

    availability = info.get("Availability", "") or get_text(
        soup.select_one("p.availability")
    )

    image_tag = soup.select_one("div.item.active img") or soup.select_one("#product_gallery img")
    image_url = urljoin(book_url, image_tag["src"]) if image_tag else ""

    return {
        "title": get_text(soup.select_one("div.product_main h1")),
        "price": to_float(get_text(soup.select_one("p.price_color"))),
        "rating": rating,
        "availability": availability,
        "stock_quantity": to_int(availability),
        "description": description,
        "category": category,
        "upc": info.get("UPC", ""),
        "product_type": info.get("Product Type", ""),
        "price_excl_tax": to_float(info.get("Price (excl. tax)")),
        "price_incl_tax": to_float(info.get("Price (incl. tax)")),
        "tax": to_float(info.get("Tax")),
        "review_count": to_int(info.get("Number of reviews")),
        "image_url": image_url,
        "product_url": book_url,
    }



# Step 4: run everything and store the dataset

def main():
    book_urls = collect_book_urls()

    records, failed = [], []
    for index, url in enumerate(book_urls, start=1):
        try:
            records.append(get_book_details(url))
        except Exception as exc:  # keep going if a single page fails
            failed.append(url)
            print(f"  FAILED: {url} ({exc})")

        if index % 50 == 0 or index == len(book_urls):
            print(f"Scraped {index}/{len(book_urls)} book pages")
        time.sleep(DELAY_SECONDS)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_excel(OUTPUT_XLSX, index=False)

    print("\n================ SUMMARY ================")
    print(f"Books saved        : {len(df)}")
    print(f"Failed pages       : {len(failed)}")
    print(f"Categories         : {df['category'].nunique()}")
    print(f"Average price (GBP): {df['price'].mean():.2f}")
    print(f"Missing descriptions: {(df['description'] == '').sum()}")
    print(f"Files created      : {OUTPUT_CSV}, {OUTPUT_XLSX}")
    print("\nPreview:")
    print(df[["title", "price", "rating", "category", "review_count"]].head())

    if failed:
        pd.Series(failed, name="failed_url").to_csv("failed_urls.csv", index=False)
        print("Failed URLs saved to failed_urls.csv (re-run to retry).")


if __name__ == "__main__":
    main()
