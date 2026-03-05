import asyncio
import pandas as pd
import os
from playwright.async_api import async_playwright

async def scrape_prothom_alo_full(max_news=1000):
    file_path = "data/prothom_alo_bangladesh_real.csv"
    existing_df = pd.DataFrame()
    existing_links = set()

    # 1. Age theke thaka data load kora (Incremental logic)
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if 'URL' in existing_df.columns:
                existing_links = set(existing_df['URL'].dropna().tolist())
                print(f"Purono file-e {len(existing_links)} ti news pawa geche. Egulo skip kora hobe.")
        except Exception as e:
            print(f"File load korte somoshya: {e}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        page = await context.new_page()
        page.set_default_timeout(60000)

        print("Bangladesh section-e jacchi...")
        try:
            await page.goto("https://www.prothomalo.com/bangladesh", wait_until="domcontentloaded", timeout=90000)
        except Exception as e:
            print(f"Page load failed: {e}")
            await browser.close()
            return

        loaded_links = set()
        print("Notun news link khoja hochhe...")

        # 2. Link collection loop
        while len(loaded_links) < max_news:
            await page.mouse.wheel(0, 5000)
            await asyncio.sleep(2)

            try:
                load_more = page.get_by_text("আরও", exact=True)
                if await load_more.is_visible():
                    await load_more.click()
                    await asyncio.sleep(3)
                else:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(2)
            except:
                pass

            links_on_page = await page.evaluate("""() => {
                const anchors = Array.from(document.querySelectorAll('h3 a.title-link'));
                return anchors.map(a => a.href);
            }""")

            new_links_found = 0
            for l in links_on_page:
                if "/bangladesh/" in l and l not in existing_links:
                    if l not in loaded_links:
                        loaded_links.add(l)
                        new_links_found += 1

            print(f"Total notun link pawa geche: {len(loaded_links)}")

            if len(loaded_links) >= max_news or (len(links_on_page) > 0 and new_links_found == 0):
                print("Ar kono notun link pawa jachhe na.")
                break

        unique_new_links = list(loaded_links)[:max_news]
        new_news_dataset = []

        # 3. Shudhu Notun News Extract kora
        for index, link in enumerate(unique_new_links):
            try:
                print(f"[{index + 1}/{len(unique_new_links)}] Scraping NEW: {link}")
                await page.goto(link, wait_until="domcontentloaded", timeout=60000)

                title = await page.inner_text("h1") if await page.query_selector("h1") else "N/A"
                paragraphs = await page.locator(".story-element-text p").all_inner_texts()
                body_content = "\n".join(paragraphs).strip()
                author = await page.inner_text("span.contributor-name") if await page.query_selector("span.contributor-name") else "Staff"
                date_time = await page.get_attribute("time", "datetime") if await page.query_selector("time") else "N/A"

                new_news_dataset.append({
                    "Title": title,
                    "Category": "Bangladesh",
                    "Author": author,
                    "Date_Time": date_time,
                    "Body": body_content,
                    "URL": link,
                    "Label": "Real"
                })
            except Exception as e:
                print(f"Skipping {link}")

        await browser.close()

        # 4. Save Logic
        if new_news_dataset:
            if not os.path.exists('data'): os.makedirs('data')
            new_df = pd.DataFrame(new_news_dataset)
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            final_df.drop_duplicates(subset=['URL'], keep='first', inplace=True)
            final_df.to_csv(file_path, index=False, encoding="utf-8-sig")
            print(f"\nSUCCESS! {len(new_news_dataset)} ti news add hoyeche. Total: {len(final_df)}")
        else:
            print("\nKono notun news pawa jayni.")

if __name__ == "__main__":
    asyncio.run(scrape_prothom_alo_full(max_news=600))