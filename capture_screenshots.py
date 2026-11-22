"""
Capture 4 High-Resolution Screenshots of Upgraded Medical-FinTech UI
DocuClaim | Team PromptCraft | PE-B13-57
"""

import time
import os
from playwright.sync_api import sync_playwright

def capture_all_views():
    os.makedirs("screenshots", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Desktop resolution 1440x900, 2x scale for crisp retina display
        context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # Wait for top header bar to appear
        page.wait_for_selector(".top-header-bar", timeout=30000)
        print("Page loaded successfully.")

        # --- VIEW 1: Dashboard (Triage & Dispute) ---
        print("Capturing View 1: Dashboard...")
        page.locator("[data-testid='stSidebar'] label:has-text('Dashboard')").click()
        time.sleep(2)
        
        # Expand Grievance Appeal Letter so it is visible
        expander = page.locator("details summary:has-text('View Formal Grievance Appeal Letter')")
        if expander.count() > 0:
            expander.first.click()
            time.sleep(1)

        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        page.screenshot(path="screenshots/01_dashboard.png", full_page=False)
        print("Saved screenshots/01_dashboard.png")

        # --- VIEW 2: Semantic Clause Retriever ---
        print("Capturing View 2: Semantic Clause Retriever...")
        page.locator("[data-testid='stSidebar'] label:has-text('Semantic Clause Retriever')").click()
        time.sleep(2)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        page.screenshot(path="screenshots/02_rag_retriever.png", full_page=False)
        print("Saved screenshots/02_rag_retriever.png")

        # --- VIEW 3: Prompt Chain Inspector ---
        print("Capturing View 3: Prompt Chain Inspector...")
        page.locator("[data-testid='stSidebar'] label:has-text('Prompt Chain Inspector')").click()
        time.sleep(2)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        page.screenshot(path="screenshots/03_prompt_inspector.png", full_page=False)
        print("Saved screenshots/03_prompt_inspector.png")

        # --- VIEW 4: Benchmark Evaluation ---
        print("Capturing View 4: Benchmark Evaluation...")
        page.locator("[data-testid='stSidebar'] label:has-text('Benchmark Evaluation')").click()
        time.sleep(2)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        page.screenshot(path="screenshots/04_benchmark_eval.png", full_page=False)
        print("Saved screenshots/04_benchmark_eval.png")

        browser.close()
        print("All 4 fresh screenshots captured successfully!")

if __name__ == "__main__":
    capture_all_views()
