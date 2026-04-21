"""
Scraper for latestrounds.com — AI startup funding data.

Strategy
--------
1. Category page (/categories/ai):
   Server-side rendered → scraped with fast plain requests + BeautifulSoup.

2. Company detail page (/companies/{slug}):
   JavaScript-rendered (React streaming) → scraped with Playwright (real
   Chromium browser). Confirmed HTML structure from live browser inspection.

Outputs:
  latestrounds_company_list.csv (comprehensive variables)
  latestrounds_funding_rounds.csv (timeline)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PWTimeout

# ── Setup ────────────────────────────────────────────────────────────────────
Path("data/raw").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://latestrounds.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://latestrounds.com",
}

PRIORITIZED_CATEGORY_LABELS = [
    "Artificial Intelligence",
    "AI",
    "Machine Learning",
    "Generative AI",
    "AI Infrastructure",
    "Enterprise AI",
    "Conversational AI",
    "Voice AI",
    "Computer Vision",
    "Robotics",
    "Automation",
    "Developer Tools",
    "Open Source",
    "API",
    "Data Engineering",
    "Databases",
    "DevOps",
    "Infrastructure Software",
    "Cloud Infrastructure",
    "Semiconductors",
    "Hardware",
    "Enterprise Software",
    "SaaS",
    "Fintech",
    "Payments",
    "Financial Services",
    "Cybersecurity",
    "Cloud Security",
    "Healthtech",
    "Healthcare",
    "Biotechnology",
    "Digital Health",
    "Climate Tech",
    "Energy",
    "Manufacturing",
    "Logistics",
    "Supply Chain Technology",
    "Edtech",
    "Legal Tech",
    "Retail Tech",
    "E-Commerce",
    "Defense Tech",
    "Space Technology",
]

CATEGORY_SLUG_OVERRIDES = {
    "AI": "ai",
    "API": "api",
    "SaaS": "saas",
    "E-Commerce": "e-commerce",
}

OVERLAPPING_LABEL_CANONICAL = {
    "ai": "Artificial Intelligence",
    "artificial intelligence ai": "Artificial Intelligence",
    "health tech": "Healthtech",
    "legaltech": "Legal Tech",
    "retailtech": "Retail Tech",
    "climatetech": "Climate Tech",
    "clean technology": "Climate Tech",
    "clean tech": "Climate Tech",
    "cleantech": "Climate Tech",
    "ecommerce": "E-Commerce",
}

TOP_TIER_INVESTORS = [
    "sequoia", "andreessen horowitz", "a16z", "accel", "tiger global", 
    "y combinator", "greylock", "kleiner perkins", "benchmark", 
    "general catalyst", "khosla", "lightspeed", "bessemer", "insight partners", 
    "index ventures", "nea", "founders fund", "coatue"
]


def _normalize_label(label: str) -> str:
    label = (label or "").lower()
    label = label.replace("&", " and ")
    label = re.sub(r"[^a-z0-9]+", " ", label)
    return re.sub(r"\s+", " ", label).strip()


def canonicalize_category_label(label: str) -> str:
    normalized = _normalize_label(label)
    return OVERLAPPING_LABEL_CANONICAL.get(normalized, label)


def slugify_category_label(label: str) -> str:
    if label in CATEGORY_SLUG_OVERRIDES:
        return CATEGORY_SLUG_OVERRIDES[label]
    slug = label.lower()
    slug = slug.replace("&", " and ")
    slug = re.sub(r"[’'`]", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


def build_category_configs() -> List[Dict[str, str]]:
    return [
        {
            "label": label,
            "slug": slugify_category_label(label),
            "canonical_label": canonicalize_category_label(label),
        }
        for label in PRIORITIZED_CATEGORY_LABELS
    ]


def build_canonical_category_priority(category_configs: List[Dict[str, str]]) -> Dict[str, int]:
    priority: Dict[str, int] = {}
    for i, cfg in enumerate(category_configs):
        canonical_label = cfg["canonical_label"]
        if canonical_label not in priority:
            priority[canonical_label] = i
    return priority


CATEGORY_CONFIGS = build_category_configs()
CANONICAL_CATEGORY_PRIORITY = build_canonical_category_priority(CATEGORY_CONFIGS)

def _get(url: str, session: requests.Session, retries: int = 3) -> Optional[BeautifulSoup]:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 429:
                wait = 15 * attempt
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(3 * attempt)
    return None

def _pw_get_html(page: Page, url: str, wait_ms: int = 3000) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except PWTimeout:
        pass
    page.wait_for_timeout(wait_ms)
    return page.content()

def _new_pw_page(browser: Browser) -> Page:
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=HEADERS["User-Agent"],
        locale="en-US",
    )
    ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return ctx.new_page()

def _delay():
    time.sleep(random.uniform(1.5, 3.0))

def parse_amount(s: str) -> Optional[float]:
    if not s or "undisclosed" in str(s).lower():
        return None
    clean = re.sub(r"[$,\s]", "", str(s).upper())
    m = re.match(r"([\d\.]+)(M|B|K)?", clean)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "B":
        return val * 1000
    if unit == "K":
        return val / 1000
    return val


def dedupe_companies_by_slug(companies: List[Dict]) -> Tuple[List[Dict], int]:
    merged: Dict[str, Dict] = {}
    duplicate_hits = 0

    for company in companies:
        slug = (company.get("slug") or "").strip()
        if not slug:
            continue

        if slug not in merged:
            base = company.copy()
            base["_matched_categories"] = set()
            base["_matched_canonical_categories"] = set()
            if base.get("matched_categories"):
                base["_matched_categories"].add(base["matched_categories"])
            if base.get("matched_canonical_categories"):
                base["_matched_canonical_categories"].add(base["matched_canonical_categories"])
            merged[slug] = base
            continue

        duplicate_hits += 1
        existing = merged[slug]
        if company.get("matched_categories"):
            existing["_matched_categories"].add(company["matched_categories"])
        if company.get("matched_canonical_categories"):
            existing["_matched_canonical_categories"].add(company["matched_canonical_categories"])

        for key, value in company.items():
            if key in {"matched_categories", "matched_canonical_categories"}:
                continue
            if not existing.get(key) and value:
                existing[key] = value

    deduped_companies: List[Dict] = []
    for company in merged.values():
        raw_labels = sorted(x for x in company.pop("_matched_categories", set()) if x)
        canonical_labels = sorted(
            (x for x in company.pop("_matched_canonical_categories", set()) if x),
            key=lambda x: CANONICAL_CATEGORY_PRIORITY.get(x, 10_000),
        )
        if canonical_labels:
            company["category"] = canonical_labels[0]
        company["matched_categories"] = ", ".join(raw_labels)
        company["matched_canonical_categories"] = ", ".join(canonical_labels)
        deduped_companies.append(company)

    return deduped_companies, duplicate_hits


def dedupe_rounds(rows: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    unique_rows: List[Dict] = []
    for row in rows:
        key = (
            (row.get("company_name") or "").strip().lower(),
            (row.get("round_type") or "").strip().lower(),
            (row.get("amount") or "").strip().lower(),
            (row.get("date") or "").strip(),
            (row.get("investors") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows, len(rows) - len(unique_rows)

def scrape_category(category_label: str, category_slug: str, canonical_label: str, session: requests.Session) -> List[Dict]:
    url = f"{BASE_URL}/categories/{category_slug}"
    soup = _get(url, session)
    if not soup:
        logger.warning(f"Failed to load category '{category_label}' from {url}")
        return []

    companies = []
    for row in soup.select("a[href^='/companies/']"):
        href = row.get("href", "")
        slug = href.split("/companies/")[-1].strip("/")
        if not slug:
            continue

        name_el = row.select_one("span.font-semibold")
        name = name_el.get_text(strip=True) if name_el else slug

        # Default properties to insert into schema
        companies.append({
            "name": name,
            "slug": slug,
            "crunchbase_url": "",
            "founded_date": "",
            "status": "Active",
            "headquarters_city": "",
            "headquarters_country": "",
            "latitude": "",
            "longitude": "",
            "category": canonical_label,
            "total_funding_usd": "",
            "last_funding_date": "",
            "last_funding_type": "",
            "valuation_usd": "",
            "num_founders": "",
            "founder_backgrounds": "",
            "serial_entrepreneur": "",
            "employee_count": "",
            "github_stars": "",
            "num_investors": "",
            "top_tier_investor": "N",
            "investor_names": "",
            "matched_categories": category_label,
            "matched_canonical_categories": canonical_label,
            "latestrounds_url": f"{BASE_URL}/companies/{slug}"
        })
    return companies

def scrape_company(slug: str, company_name: str, pw_page: Page) -> Tuple[List[Dict], Dict]:
    url = f"{BASE_URL}/companies/{slug}"
    rounds = []
    meta = {}

    try:
        html = _pw_get_html(pw_page, url)
        soup = BeautifulSoup(html, "html.parser")
        text_piped = soup.get_text(separator="|", strip=True)

        # Extract Company Metadata
        m_founded = re.search(r"Founded\|(\d{4})", text_piped)
        if m_founded:
            meta["founded_date"] = m_founded.group(1)

        m_emp = re.search(r"Employees\|([\d,\+]+)", text_piped)
        if m_emp:
            meta["employee_count"] = m_emp.group(1).replace(",", "")

        m_loc = re.search(r"Location\|([^|]+)", text_piped)
        if m_loc:
            full_loc = m_loc.group(1).strip()
            parts = full_loc.rsplit(",", 1)
            if len(parts) == 2:
                meta["headquarters_city"] = parts[0].strip()
                meta["headquarters_country"] = parts[1].strip()
            else:
                meta["headquarters_country"] = full_loc

        m_val = re.search(r"Valuation\|(\$[\d,\.]+[MBK]?)", text_piped)
        if m_val:
            meta["valuation_usd"] = m_val.group(1)

        m_raised = re.search(r"Total Raised\|(\$[\d,\.]+[MBK]?)", text_piped)
        if m_raised:
            meta["total_funding_usd"] = m_raised.group(1)

        m_ind = re.search(r"Industry\|(.+?)\|Funding Timeline", text_piped)
        if m_ind:
            industries = m_ind.group(1).split('|')
            meta["category"] = ", ".join(industries)

        m_inv_cnt = re.search(r"Unique investors\|(\d+)", text_piped)
        if m_inv_cnt:
            meta["num_investors"] = m_inv_cnt.group(1)

        # Extract Rounds
        round_blocks = soup.find_all("div", class_=lambda c: c and "flex-1" in c and "min-w-0" in c and "pb-2" in c)
        
        all_investors = set()
        
        for block in round_blocks:
            full_text = block.get_text(separator=" ", strip=True)
            amount_m = re.search(r"(\$[\d,\.]+[MBK]?B?|Undisclosed)\s+raised in\s+([A-Z0-9\+\-\s]+?)(?=\s+[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4})", full_text, re.IGNORECASE)
            if amount_m:
                amount = amount_m.group(1).strip()
                round_type = amount_m.group(2).strip()
            else:
                amount_m = re.search(r"(\$[\d,\.]+[MBK]?B?|Undisclosed)\s+raised in\s+(.+?)(?=\s|$)", full_text, re.IGNORECASE)
                if not amount_m:
                    continue
                amount = amount_m.group(1).strip()
                round_type = amount_m.group(2).strip()
                
            date_m = re.search(r"([A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4})", full_text)
            date = date_m.group(1) if date_m else ""

            investor_links = block.find_all("a", href=re.compile(r"^/investors/"))
            round_investors = [a.get_text(strip=True) for a in investor_links]
            all_investors.update(round_investors)

            rounds.append({
                "company_name": company_name,
                "round_type": round_type,
                "amount": amount,
                "date": date,
                "investors": ", ".join(round_investors),
            })
            
        if rounds:
            # Sort rounds by date approx if possible, or just take first as latest
            meta["last_funding_date"] = rounds[0]["date"]
            meta["last_funding_type"] = rounds[0]["round_type"]

        inv_str = ", ".join(list(all_investors))
        meta["investor_names"] = inv_str
        
        has_top_tier = any(tti in inv_str.lower() for tti in TOP_TIER_INVESTORS)
        meta["top_tier_investor"] = "Y" if has_top_tier else "N"

    except Exception as e:
        logger.error(f"  Failed to scrape {slug}: {e}")

    return rounds, meta

def main():
    logger.info("=" * 60)
    logger.info("LatestRounds.com Comprehensive AI Scraper")
    logger.info("=" * 60)
    session = requests.Session()

    all_companies: List[Dict] = []
    logger.info(f"Scraping {len(CATEGORY_CONFIGS)} prioritized categories...")
    for cat_cfg in CATEGORY_CONFIGS:
        logger.info(f"Category: {cat_cfg['label']} ({cat_cfg['slug']})")
        category_companies = scrape_category(
            cat_cfg["label"],
            cat_cfg["slug"],
            cat_cfg["canonical_label"],
            session,
        )
        all_companies.extend(category_companies)
        logger.info(f"  → discovered {len(category_companies)} company links")
        _delay()

    unique, overlap_duplicate_count = dedupe_companies_by_slug(all_companies)
    logger.info(
        f"Category overlap dedupe removed {overlap_duplicate_count} duplicates; {len(unique)} unique companies remain."
    )

    if not unique:
        logger.error("No companies found.")
        return

    all_rounds: List[Dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        pw_page = _new_pw_page(browser)

        for i, company in enumerate(unique, 1):
            slug = company["slug"]
            name = company["name"] or slug
            logger.info(f"  [{i}/{len(unique)}] {name}")

            rounds, meta = scrape_company(slug, name, pw_page)
            
            # Update company dictionary with scraped metadata
            company.update(meta)
            
            if rounds:
                all_rounds.extend(rounds)
                logger.info(f"    → {len(rounds)} round(s) found.")
            
            _delay()
        browser.close()

    all_rounds, duplicate_round_rows = dedupe_rounds(all_rounds)
    if duplicate_round_rows:
        logger.info(f"Removed {duplicate_round_rows} duplicate funding round rows.")
    df_companies = pd.DataFrame(unique)
    
    # Parse financial fields to float
    def safe_parse(x):
        try:
            val = parse_amount(x)
            return val * 1_000_000 if val else ""
        except:
            return ""
            
    if "total_funding_usd" in df_companies.columns:
        df_companies["total_funding_usd"] = df_companies["total_funding_usd"].apply(safe_parse)
    if "valuation_usd" in df_companies.columns:
        df_companies["valuation_usd"] = df_companies["valuation_usd"].apply(safe_parse)

    # Reorder to match exact user request
    cols = [
        "name", "crunchbase_url", "founded_date", "status",
        "headquarters_city", "headquarters_country", "latitude", "longitude",
        "category", "total_funding_usd", "last_funding_date", "last_funding_type", "valuation_usd",
        "num_founders", "founder_backgrounds", "serial_entrepreneur",
        "employee_count", "github_stars",
        "num_investors", "top_tier_investor", "investor_names"
    ]
    
    # Ensure all columns exist before selecting
    for col in cols:
        if col not in df_companies.columns:
            df_companies[col] = ""
            
    df_companies = df_companies[cols]
    df_companies.to_csv("data/raw/latestrounds_company_list.csv", index=False)

    df_rounds = pd.DataFrame(all_rounds)
    if not df_rounds.empty:
        df_rounds.to_csv("data/raw/latestrounds_funding_rounds.csv", index=False)

    logger.info("Done! Saved updated comprehensive schema to data/raw/latestrounds_company_list.csv")

if __name__ == "__main__":
    main()
