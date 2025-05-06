import asyncio
import time
import yaml # type: ignore
import re
import urllib3 # type: ignore
import requests # type: ignore
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString # type: ignore
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError # type: ignore

# ─── Suppress insecure‐request warnings ───
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Configuration ───
SITES = [
    {"name": "MHF_UK_Best_Tips",     "url": "https://www.mentalhealth.org.uk/explore-mental-health/publications/our-best-mental-health-tips",       "selector": "main"},
    {"name": "NIMH_Caring",          "url": "https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health",                             "selector": "main"},
    {"name": "Karpagam_Stress_Tips", "url": "https://karpagamhospital.in/10-mental-health-tips-for-stress-relief/",                              "selector": "article"},
    {"name": "Telemanas_Helplines",  "url": "https://telemanas.mohfw.gov.in/",                                                                  "selector": None},
    {"name": "FindaHelpline_Suicide","url": "https://findahelpline.com/countries/in/topics/suicidal-thoughts",                           "selector": "div.views-row"},
    {"name": "WHO_Mental_Health",    "url": "https://www.who.int/health-topics/mental-health",                                               "selector": "main"},
    {"name": "Cleveland_Depression", "url": "https://my.clevelandclinic.org/health/diseases/9290-depression",                                  "selector": "main"},
    {"name": "MayoClinic_Stress",    "url": "https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress-relievers/art-20047257", "selector": "main"},
    {"name": "Healthline_Stress",    "url": "https://www.healthline.com/nutrition/16-ways-relieve-stress-anxiety",                         "selector": "main"},
]

OUTPUT_DIR = Path("mental_health_Knowledge-Base")
OUTPUT_DIR.mkdir(exist_ok=True)

PHONE_REGEX = r"\+?\d[\d\-\s]{7,}\d"

# ─── Cleanup & Markdown Conversion ───
def clean_boilerplate(soup):
    for tag in soup.select("script, style, nav, header, footer, aside, noscript"):
        tag.decompose()

def element_to_markdown(elem) -> str:
    lines = []
    for child in elem.children:
        if isinstance(child, NavigableString):
            continue
        t = child.name.lower()
        text = child.get_text(strip=True)
        if not text:
            continue
        if t in [f"h{i}" for i in range(1,7)]:
            lines.append(f"{'#'*int(t[1])} {text}")
        elif t == "p":
            lines.append(text)
        elif t in ("ul","ol"):
            ordered = (t == "ol")
            for idx, li in enumerate(child.find_all("li", recursive=False), 1):
                prefix = f"{idx}." if ordered else "-"
                lines.append(f"{prefix} {li.get_text(strip=True)}")
        elif t == "a":
            href = child.get("href","").strip()
            lines.append(f"[{text}]({href})" if href else text)
        else:
            sub = element_to_markdown(child)
            if sub:
                lines.append(sub)
    return "\n\n".join(lines)

# ─── Fallback scraper using requests ───
def scrape_with_requests(url, selector=None):
    resp = requests.get(url, timeout=15, verify=False)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    clean_boilerplate(soup)
    container = selector and soup.select_one(selector) or soup.select_one("main") or soup.body
    if not container:
        container = soup.body
    return element_to_markdown(container)

# ─── Main async scraper ───
async def scrape_and_clean():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for site in SITES:
            name, url, sel = site["name"], site["url"], site["selector"]
            print(f"→ Scraping {name} …")
            md_body = ""
            used_fallback = False

            if name == "Telemanas_Helplines":
                # Telemanas: only phone numbers via regex
                used_fallback = True
                html = requests.get(url, timeout=10, verify=False).text
                phones = set(re.findall(PHONE_REGEX, html))
                md_body = "\n".join(f"- {ph}" for ph in sorted(phones))

            else:
                # Try Playwright first
                try:
                    await page.goto(url, timeout=15000)
                    await page.wait_for_timeout(3000)
                    html = await page.content()
                    soup = BeautifulSoup(html, "html.parser")
                    clean_boilerplate(soup)

                    container = sel and soup.select_one(sel) or soup.select_one("main")
                    if not container:
                        raise ValueError(f"selector `{sel}` not found")
                    md_body = element_to_markdown(container)

                except (PlaywrightTimeoutError, Exception) as e:
                    print(f"   [Info] Playwright failed ({e}); falling back to requests")
                    used_fallback = True
                    try:
                        md_body = scrape_with_requests(url, selector=sel)
                    except Exception as re_err:
                        print(f"   [Error] Requests fallback failed: {re_err}")
                        md_body = ""

            # Write file if we got anything
            if md_body.strip():
                fm = {"source": url, "scraped_at": time.strftime("%Y-%m-%d"), "title": name}
                full_md = f"---\n{yaml.safe_dump(fm)}---\n\n{md_body.strip()}"
                out_file = OUTPUT_DIR / f"{name}.md"
                out_file.write_text(full_md, encoding="utf-8")
                branch = "requests" if used_fallback else "playwright"
                print(f"   ✔ Saved via {branch} → {out_file.name}")
            else:
                print(f"   [Warning] No content extracted for {name}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_and_clean())
