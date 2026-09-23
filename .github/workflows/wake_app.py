import os
import re
from playwright.sync_api import sync_playwright

APP_URL = os.environ["APP_URL"]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(APP_URL, wait_until="networkidle", timeout=60000)

    wake_button = page.get_by_text(re.compile("get this app back up", re.I))
    if wake_button.count() > 0:
        wake_button.first.click()
        print("App was asleep — clicked wake button.")
        page.wait_for_timeout(15000)
    else:
        print("App was already awake.")

    browser.close()
