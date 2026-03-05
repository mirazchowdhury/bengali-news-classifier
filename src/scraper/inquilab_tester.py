import asyncio
import pandas as pd
import os
from playwright.async_api import async_playwright


async def scrape_ittefaq(max_news=200):
    file_path = "data/ittefaq_real_news.csv"

    async with async_playwright() as p:
        # Load More check korar jonno headless=False rakha valo debug korar somoy
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        page = await context.new_page()

        print("Ittefaq National section-e jacchi...")
        await page.goto("https://www.ittefaq.com.bd/national", wait_until="networkidle")

        # --- PHASE 1: LOAD MORE & LINK COLLECTION ---
        all_links = set()
        print(f"{max_news} ti news link collect kora hochhe...")

        while len(all_links) < max_news:
            # Current page theke links collect kora
            current_links = await page.evaluate("""() => {
                const anchors = Array.from(document.querySelectorAll('.tag_title_holder h2.title a'));
                return anchors.map(a => a.href);
            }""")

            for link in current_links:
                all_links.add(link)

            print(f"Collected links: {len(all_links)}")

            if len(all_links) >= max_news:
                break

            # "আরও" (Load More) button handle kora
            try:
                # Selector: class jeta 'ajax_load_btn' hold kore
                load_more_btn = page.locator(".ajax_load_btn:has-text('আরও')")

                if await load_more_btn.is_visible():
                    print("Clicking 'Load More' button...")
                    await load_more_btn.click()
                    # AJAX load hobar jonno wait kora
                    await asyncio.sleep(3)
                else:
                    print("No more 'Load More' button found.")
                    break
            except Exception as e:
                print(f"Load more e somoshya: {e}")
                break

        final_links = list(all_links)[:max_news]
        dataset = []

        # --- PHASE 2: DATA EXTRACTION ---
        for index, link in enumerate(final_links):
            try:
                print(f"[{index + 1}/{len(final_links)}] Scraping: {link}")
                await page.goto(link, wait_until="domcontentloaded", timeout=60000)

                title = await page.inner_text("h1") if await page.query_selector("h1") else "N/A"

                date_element = await page.query_selector("span[itemprop='datePublished']")
                date_time = await date_element.get_attribute("content") if date_element else "N/A"

                img_element = await page.query_selector(".featured_image img")
                img_url = await img_element.get_attribute("src") if img_element else "N/A"
                if img_url and img_url.startswith("//"): img_url = "https:" + img_url

                caption = await page.inner_text(".jw_media_caption") if await page.query_selector(
                    ".jw_media_caption") else "N/A"

                paragraphs = await page.locator("div[itemprop='articleBody'] p").all_inner_texts()
                body_content = "\n".join([p.strip() for p in paragraphs if "Google News" not in p]).strip()

                dataset.append({
                    "Title": title,
                    "Date_Time": date_time,
                    "Image_URL": img_url,
                    "Image_Caption": caption,
                    "Body": body_content,
                    "URL": link
                })

            except Exception as e:
                print(f"Error scraping {link}: {e}")
                continue

        await browser.close()

        # --- PHASE 3: SAVE TO CSV ---
        if dataset:
            if not os.path.exists('data'): os.makedirs('data')
            df = pd.DataFrame(dataset)

            if os.path.exists(file_path):
                old_df = pd.read_csv(file_path)
                df = pd.concat([old_df, df]).drop_duplicates(subset=['URL'])

            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            print(f"SUCCESS! {len(dataset)} ti news save kora hoyeche.")
        else:
            print("Kono data scrap kora jayni.")


if __name__ == "__main__":
    asyncio.run(scrape_ittefaq(max_news=200))