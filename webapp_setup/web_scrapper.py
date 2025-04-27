import asyncio
from pathlib import Path
from bs4 import BeautifulSoup # type: ignore
from playwright.async_api import async_playwright # type: ignore
import yaml # type: ignore
import time

# Same SITES list but without CSS selector issues yet
SITES = [
    # Mental Health Tips
    {
        "name": "MHF_UK_Best_Tips",
        "url": "https://www.mentalhealth.org.uk/explore-mental-health/publications/our-best-mental-health-tips",
        "selector": "main",
    },
    {
        "name": "NIMH_Caring",
        "url": "https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health",
        "selector": "main",
    },
    {
        "name": "Karpagam_Stress_Tips",
        "url": "https://karpagamhospital.in/10-mental-health-tips-for-stress-relief/",
        "selector": "div.elementor-widget-container",
    },
    # Crisis
    {
        "name": "Telemanas_Helplines",
        "url": "https://telemanas.mohfw.gov.in/",
        "selector": "body",   # fallback
    },
    {
        "name": "FindaHelpline_Suicide",
        "url": "https://findahelpline.com/countries/in/topics/suicidal-thoughts",
        "selector": "body",   # fallback
    },
    # Resources
    {
        "name": "WHO_Mental_Health",
        "url": "https://www.who.int/health-topics/mental-health",
        "selector": "section.article-body",
    },
    {
        "name": "Cleveland_Depression",
        "url": "https://my.clevelandclinic.org/health/diseases/9290-depression",
        "selector": "div.main-content",
    },
    {
        "name": "MayoClinic_Stress",
        "url": "https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress-relievers/art-20047257",
        "selector": "div.content",
    },
    {
        "name": "Healthline_Stress_Anxiety",
        "url": "https://www.healthline.com/nutrition/16-ways-relieve-stress-anxiety",
        "selector": "div.css-0",
    },
]

OUTPUT_DIR = Path("mental_health_docs")
OUTPUT_DIR.mkdir(exist_ok=True)

async def scrape_sites():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for site in SITES:
            print(f"→ Scraping {site['name']} from {site['url']} ...")
            try:
                await page.goto(site["url"], timeout=60000)
                await page.wait_for_timeout(5000)  # wait 5s for JS to load

                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")

                element = soup.select_one(site["selector"])
                if not element:
                    print(f"   [Warning] Selector {site['selector']} not found. Saving full page instead.")
                    text = soup.get_text(separator="\n\n")
                else:
                    text = element.get_text(separator="\n\n")
                
                # Save as markdown with frontmatter
                frontmatter = {
                    "source": site["url"],
                    "scraped_at": time.strftime("%Y-%m-%d"),
                    "title": site["name"],
                }
                md = f"---\n{yaml.safe_dump(frontmatter)}---\n\n{text.strip()}"
                out_path = OUTPUT_DIR / f"{site['name']}.md"
                out_path.write_text(md, encoding="utf-8")
                print(f"   ✔ Saved {out_path}")

            except Exception as e:
                print(f"   [Error] {e}")

        await browser.close()

asyncio.run(scrape_sites())
