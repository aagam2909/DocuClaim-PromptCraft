"""
Automated Headless UI Screenshot Capture for DocuClaim
Team PromptCraft (PE-B13-57)
Aagam Jain (S24CSEU0263), Riya Tyagi (S24CSEU0280), Shubhashni Kant (S24CSEU0535)
"""

import time
import os
from playwright.sync_api import sync_playwright

def capture_ui_screenshots():
    os.makedirs("screenshots", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # Wait for Streamlit to load hero banner
        page.wait_for_selector(".hero-banner", timeout=30000)
        print("Page loaded successfully.")

        # Click the 'Run DocuClaim Audit' button
        print("Clicking 'Run DocuClaim Audit' button...")
        audit_btn = page.locator("button:has-text('Run DocuClaim Audit')")
        if audit_btn.count() > 0:
            audit_btn.first.click()
            time.sleep(4)
            page.wait_for_load_state("networkidle")
        else:
            print("Audit button not found, checking alternatives...")

        # Also open the intermediate JSON expander so it's visible if needed
        expander = page.locator("details summary:has-text('View Intermediate Prompt 1 & Prompt 2 JSON Outputs')")
        if expander.count() > 0:
            expander.first.click()
            time.sleep(1)

        print("Capturing 4 high-resolution modular screenshots...")

        # Screenshot 1: Module 1 (Hero + Input Loader)
        # Scroll to top
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        page.screenshot(path="screenshots/01_input_loader.png", full_page=False)
        print("Saved screenshots/01_input_loader.png")

        # Screenshot 2: Module 2 (Semantic Clause Retriever)
        # Scroll to Module 2
        page.evaluate("""() => {
            const el = document.querySelectorAll('.module-header')[1];
            if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
        }""")
        time.sleep(1)
        page.screenshot(path="screenshots/02_rag_retriever.png", full_page=False)
        print("Saved screenshots/02_rag_retriever.png")

        # Screenshot 3: Module 3 (Multi-Stage Deduction & Proportionate Math Auditor)
        page.evaluate("""() => {
            const el = document.querySelectorAll('.module-header')[2];
            if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
        }""")
        time.sleep(1)
        page.screenshot(path="screenshots/03_deduction_audit.png", full_page=False)
        print("Saved screenshots/03_deduction_audit.png")

        # Screenshot 4: Module 4 (Plain-English Explainer & Appeal Letter)
        page.evaluate("""() => {
            const el = document.querySelectorAll('.module-header')[3];
            if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
        }""")
        time.sleep(1)
        page.screenshot(path="screenshots/04_appeal_generator.png", full_page=False)
        print("Saved screenshots/04_appeal_generator.png")

        browser.close()
        print("All 4 screenshots captured successfully!")

if __name__ == "__main__":
    capture_ui_screenshots()
