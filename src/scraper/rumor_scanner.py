import asyncio
import pandas as pd
import os
from playwright.async_api import async_playwright


async def scrape_rumor_scanner_incremental(target_count=500):
    file_path = "data/rumor_data_fake.csv"
    existing_df = pd.DataFrame()
    existing_links = set()

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if 'URL' in existing_df.columns:
                existing_links = set(existing_df['URL'].dropna().tolist())
                print(f"Existing links: {len(existing_links)}")
        except:
            pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        page = await context.new_page()
        page.set_default_timeout(90000)

        print("Navigating to Rumor Scanner...")
        await page.goto("https://rumorscanner.com", wait_until="domcontentloaded")

        all_new_links = []
        retry_count = 0

        # --- PHASE 1: LINK COLLECTION ---
        while len(all_new_links) < target_count and retry_count < 20:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

            links_on_page = await page.evaluate("""() => {
                const anchors = Array.from(document.querySelectorAll('h3.entry-title a'));
                return anchors.map(a => a.href);
            }""")

            for l in links_on_page:
                if l not in existing_links and l not in all_new_links:
                    all_new_links.append(l)

            print(f"Total Unique New Links: {len(all_new_links)}")

            if len(all_new_links) >= target_count:
                break

            try:
                load_more_selector = ".td_ajax_load_more_js"
                if await page.is_visible(load_more_selector):
                    await page.evaluate("document.querySelector('.td_ajax_load_more_js').click()")
                    await asyncio.sleep(5)
                    retry_count = 0
                else:
                    await page.mouse.wheel(0, -500)
                    await asyncio.sleep(2)
                    retry_count += 1
            except:
                retry_count += 1

        # --- PHASE 2: DATA EXTRACTION ---
        new_rumor_dataset = []
        for index, link in enumerate(all_new_links[:target_count]):
            try:
                print(f"[{index + 1}/{len(all_new_links[:target_count])}] Scraping: {link}")
                await page.goto(link, wait_until="domcontentloaded")

                title = await page.inner_text("h1") if await page.query_selector("h1") else "N/A"
                author = await page.inner_text(".tdb-author-name") if await page.query_selector(
                    ".tdb-author-name") else "RS Team"
                date_time = await page.get_attribute("time.entry-date", "datetime") if await page.query_selector(
                    "time.entry-date") else "N/A"

                paragraphs = await page.locator(".td-post-content p").all_inner_texts()
                fake_summary = paragraphs[0] if len(paragraphs) > 0 else "N/A"
                full_desc = "\n".join(paragraphs)

                new_rumor_dataset.append({
                    "Title": title, "Author": author, "Date_Time": date_time,
                    "Body": fake_summary, "Fact_Check_Description": full_desc,
                    "URL": link, "Label": "Fake"
                })
            except:
                continue

        await browser.close()

        # --- PHASE 3: SAVE ---
        if new_rumor_dataset:
            new_df = pd.DataFrame(new_rumor_dataset)
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            final_df.drop_duplicates(subset=['URL'], keep='first', inplace=True)
            final_df.to_csv(file_path, index=False, encoding="utf-8-sig")
            print(f"SUCCESS! Dataset now has {len(final_df)} entries.")


if __name__ == "__main__":
    asyncio.run(scrape_rumor_scanner_incremental(target_count=400))