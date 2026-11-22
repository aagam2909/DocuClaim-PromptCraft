"""
DocuClaim: Automated Verification Suite for Core Adjudication Cases
Team PromptCraft | PE-B13-57
Tests:
  - Case 1: Room Rent Breach + Pharmacy Exemption (Billed ₹90k, Settled ₹45k -> Recoverable ₹18k)
  - Case 2: PED 40-Month Tenure Repudiation (Billed ₹66k, Settled ₹24k -> Recoverable ₹42k)
  - Case 3: Missing Air Ambulance Guardrail (Billed ₹109k, Settled ₹24k -> Recoverable ₹0)
  - Test 4: PDF Extraction (sample_policy.pdf)
"""

import sys
import logging
import warnings

# Suppress Streamlit bare-mode import warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pypdf
from pipeline import (
    SAMPLE_POLICY_TEXT,
    PRESETS,
    retrieve_top_chunks,
    simulate_3_stage_pipeline,
    normalize_and_verify_stage2_output
)

def run_tests():
    print("=" * 80)
    print("DOCUCLAIM ADJUDICATION ENGINE — END-TO-END VERIFICATION")
    print("Team PromptCraft | PE-B13-57")
    print("=" * 80)

    all_passed = True

    # -------------------------------------------------------------
    # CASE 1: Room Rent Breach + Wrongful Pharmacy Deduction
    # -------------------------------------------------------------
    print("\n[TEST 1] Case 1: Room Rent Breach + Wrongful Pharmacy Deduction")
    c1_preset = PRESETS["Preset 1: Room Rent Breach + Wrongful Pharmacy Deduction (Disputable)"]
    c1_chunks = retrieve_top_chunks(c1_preset["policy"], c1_preset["bill"], top_k=3)
    s1, s2, s3 = simulate_3_stage_pipeline(c1_preset["policy"], c1_preset["bill"], c1_chunks)

    billed_1 = s2["total_billed"]
    settled_1 = s2["total_settled"]
    rec_1 = s2["total_recoverable_amount"]
    pharm_audit = next((it for it in s2["line_item_audits"] if "pharmacy" in it["item"].lower()), None)
    pharm_status = pharm_audit["status"] if pharm_audit else "UNKNOWN"

    print(f"  • Total Billed: ₹{billed_1:,.0f} (Expected: ₹90,000)")
    print(f"  • Total Settled: ₹{settled_1:,.0f} (Expected: ₹45,000)")
    print(f"  • Disputed / Recoverable Amount: ₹{rec_1:,.0f} (Expected: ₹18,000)")
    print(f"  • Pharmacy Verdict: {pharm_status} per {pharm_audit.get('clause_cited') if pharm_audit else 'N/A'}")

    c1_pass = (billed_1 == 90000.0 and settled_1 == 45000.0 and rec_1 == 18000.0 and pharm_status == "INVALID")
    print(f"  >>> RESULT: {'✅ PASS' if c1_pass else '❌ FAIL'}")
    if not c1_pass:
        all_passed = False

    # -------------------------------------------------------------
    # CASE 2: Pre-Existing Disease (PED) 40-Month Tenure
    # -------------------------------------------------------------
    print("\n[TEST 2] Case 2: Pre-Existing Disease (PED) 40-Month Tenure (>36m)")
    c2_preset = PRESETS["Preset 2: Pre-Existing Disease (PED) 40-Month Tenure (Invalid Rejection)"]
    c2_chunks = retrieve_top_chunks(c2_preset["policy"], c2_preset["bill"], top_k=3)
    s1, s2, s3 = simulate_3_stage_pipeline(c2_preset["policy"], c2_preset["bill"], c2_chunks)

    billed_2 = s2["total_billed"]
    settled_2 = s2["total_settled"]
    rec_2 = s2["total_recoverable_amount"]
    surg_audit = next((it for it in s2["line_item_audits"] if any(k in it["item"].lower() for k in ["cardiac", "surgery", "stent"])), None)
    surg_status = surg_audit["status"] if surg_audit else "UNKNOWN"

    print(f"  • Total Billed: ₹{billed_2:,.0f} (Expected: ₹66,000)")
    print(f"  • Total Settled: ₹{settled_2:,.0f} (Expected: ₹24,000)")
    print(f"  • Disputed / Recoverable Amount: ₹{rec_2:,.0f} (Expected: ₹42,000)")
    print(f"  • Surgery Repudiation Verdict: {surg_status} per {surg_audit.get('clause_cited') if surg_audit else 'N/A'}")

    c2_pass = (billed_2 == 66000.0 and settled_2 == 24000.0 and rec_2 == 42000.0 and surg_status == "INVALID")
    print(f"  >>> RESULT: {'✅ PASS' if c2_pass else '❌ FAIL'}")
    if not c2_pass:
        all_passed = False

    # -------------------------------------------------------------
    # CASE 3: Missing Air Ambulance Clause Edge Case
    # -------------------------------------------------------------
    print("\n[TEST 3] Case 3: Missing Air Ambulance Clause (Insufficient Data Guardrail)")
    c3_preset = PRESETS["Preset 3: Missing Air Ambulance Clause Edge Case (Insufficient Data Guardrail)"]
    c3_chunks = retrieve_top_chunks(c3_preset["policy"], c3_preset["bill"], top_k=3)
    s1, s2, s3 = simulate_3_stage_pipeline(c3_preset["policy"], c3_preset["bill"], c3_chunks)

    billed_3 = s2["total_billed"]
    settled_3 = s2["total_settled"]
    rec_3 = s2["total_recoverable_amount"]
    amb_audit = next((it for it in s2["line_item_audits"] if "ambulance" in it["item"].lower()), None)
    amb_status = amb_audit["status"] if amb_audit else "UNKNOWN"

    print(f"  • Total Billed: ₹{billed_3:,.0f} (Expected: ₹1,09,000)")
    print(f"  • Total Settled: ₹{settled_3:,.0f} (Expected: ₹24,000)")
    print(f"  • Disputed / Recoverable Amount: ₹{rec_3:,.0f} (Expected: ₹0)")
    print(f"  • Air Ambulance Verdict: {amb_status} per {amb_audit.get('clause_cited') if amb_audit else 'N/A'}")

    c3_pass = (billed_3 == 109000.0 and settled_3 == 24000.0 and rec_3 == 0.0 and amb_status == "INSUFFICIENT_CLAUSE_DATA")
    print(f"  >>> RESULT: {'✅ PASS' if c3_pass else '❌ FAIL'}")
    if not c3_pass:
        all_passed = False

    # -------------------------------------------------------------
    # TEST 4: Policy PDF Text Extraction & Precedence Check
    # -------------------------------------------------------------
    print("\n[TEST 4] Policy PDF Text Extraction (sample_policy.pdf)")
    try:
        reader = pypdf.PdfReader("sample_policy.pdf")
        pdf_text = "".join([page.extract_text() + "\n" for page in reader.pages]).strip()
        print(f"  • Extracted {len(reader.pages)} page(s) ({len(pdf_text):,} characters)")
        has_clause4 = "Clause 4.1" in pdf_text
        print(f"  • Grounded Clause 4.1 present: {has_clause4}")
        pdf_pass = len(pdf_text) > 500 and has_clause4
        print(f"  >>> RESULT: {'✅ PASS' if pdf_pass else '❌ FAIL'}")
        if not pdf_pass:
            all_passed = False
    except Exception as e:
        print(f"  • Error reading sample_policy.pdf: {e}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED PERFECTLY (4/4)! 🚀")
    else:
        print("SOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
