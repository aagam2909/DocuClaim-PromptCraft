# DocuClaim: RAG-Driven Health Insurance Policy Triage & Dispute Assistant

**Team PromptCraft** | **Team ID: `PE-B13-57`**  
**Team Members:** Aagam Jain (`S24CSEU0263`), Riya Tyagi (`S24CSEU0280`), Shubhashni Kant (`S24CSEU0535`)  
**Course:** SCSE4041 Prompt Engineering | **Batch Group:** B13-24  
**Model:** Google `gemini-2.5-flash` via `google-genai` SDK  

---

## 📌 Executive Overview

**DocuClaim** is an enterprise-grade AI forensic audit and grievance redressal system for health insurance claims. When hospital bills are settled, Third-Party Administrators (TPAs) and insurers frequently apply **proportionate room rent deductions** across the entire bill—unlawfully deducting non-associated expenses like pharmacy, medicines, and consumables in direct contravention of policy terms and IRDAI regulations.

DocuClaim automates the forensic review process:
1. **RAG Context Grounding**: Semantically retrieves the exact policy clauses governing room rent limits, ICU caps, and pre-existing disease (PED) waiting periods.
2. **3-Stage Sequential Prompt Pipeline**: Deconstructs noisy hospital bills into structured financial entities, performs step-by-step ratio arithmetic with strict negative constraints, and enforces statutory exemptions.
3. **Dual-Audience Generation**: Translates complex forensic findings into an 8th-grade reading level plain-English summary for the patient and a formal, citation-backed legal appeal letter for the Grievance Redressal Officer (GRO).

---

## 🏗️ 4-Page Medical-FinTech Command Center Architecture

DocuClaim's interface is structured as a minimalist medical-fintech command center featuring seamless sidebar navigation across four specialized views:

```mermaid
flowchart LR
    V1["1. Dashboard (Triage & Dispute)"] --> V2["2. Semantic Clause Retriever (RAG)"]
    V2 --> V3["3. Prompt Chain Inspector (JSON Trace)"]
    V3 --> V4["4. Benchmark Evaluation (12 Cases)"]
```

### 1. Dashboard (Triage & Dispute)
- **Top Row KPI Cards**: Immediate executive visibility into `Total Billed` (e.g. ₹75,000), `Total Settled` (e.g. ₹37,500), and `Disputed / Recoverable Amount` highlighted in crimson (e.g. ₹12,500).
- **Split Ingestion & Resolution Layout**:
  - **Left Card (Document Context & Input)**: Policy PDF uploader (`pypdf` parser) with pre-filled clauses (4.1, 4.2, 7.1, 7.2, 8.4), itemized hospital bill text area, and primary **"Run Claim Audit"** trigger.
  - **Right Card (Deduction Audit & Resolution)**: Itemized forensic audit table with color-coded status badges (`VALID` emerald, `INVALID` crimson, `INSUFFICIENT_CLAUSE_DATA` amber), step-by-step ratio math explanation, 8th-grade patient plain-English summary, and expandable **Grievance Appeal Letter** editor with one-click `.txt` download (`DocuClaim_Grievance_Appeal.txt`).

### 2. Semantic Clause Retriever (RAG Context Grounding)
- **Keyword Search & Filter**: Real-time search across extracted policy clauses.
- **Top-3 Grounded Relevance Cards**: Evaluates hospital bill queries against policy chunks via TF-IDF vectorization and cosine similarity, rendering match-percentage badges (e.g. `96% Match` on Clause 4.1 Room Rent & Pharmacy Exemption) with source metadata and word counts.

### 3. Prompt Chain Inspector
- **Sequential 3-Stage Trace Cards**: Visualizes auditable handoffs with emerald `Verified` badges:
  - **Prompt 1 (Entity Extractor)**: Standardized financial JSON schema (`sum_insured`, `eligible_room_limit`, `actual_room_rent`, `line_items`).
  - **Prompt 2 (CoT Math Auditor)**: Mathematical ratio calculation (`5,000 / 10,000 = 0.50`), exemption enforcement, and audited line items.
  - **Prompt 3 (Dual-Audience Synthesizer)**: Patient explanation and formal legal grievance letter preview.

### 4. Benchmark Evaluation
- **Top Row Version Cards**: Quantifies prompt engineering progression: `v1 Baseline (50%)`, `v2 Role+Schema (67%)`, `v3 Few-Shot+Guardrails (92%)`, and `v4 Final Chained (100%)`.
- **12-Case Benchmark Table**: Renders the full `TC-01` through `TC-12` test suite loaded from `data/synthetic_test_cases.json` showing 100% `PASS` status.

---

## ⚡ Prompt Pipeline Modes: Baseline vs. Production Chained

DocuClaim features a sidebar toggle allowing evaluators to directly compare prompt engineering approaches:

| Feature | Mode A — v1: Zero-Shot Baseline | Mode B — v4: Production 3-Stage Pipeline |
| :--- | :--- | :--- |
| **Pipeline Architecture** | Single-turn zero-shot prompt | Sequential 3-stage LLM pipeline with intermediate JSON handoffs |
| **Model & Parameters** | `gemini-2.5-flash` | `gemini-2.5-flash` (`temp=0.1`, `response_mime_type="application/json"`) |
| **Entity Extraction** | None (Direct textual response) | Stage 1 isolates financial entities, line items, and tariffs |
| **Room Rent Math** | Prone to hallucinations and ratio errors | Stage 2 computes $\frac{\text{Eligible Room Limit}}{\text{Actual Room Rent}}$ step-by-step |
| **Exemption Handling** | Frequently misses pharmacy/medicine exemptions | Stage 2 prioritizes explicit exemptions before applying cuts |
| **Missing Clause Guardrail** | Guesses or invents validity | Strictly flags `INSUFFICIENT_CLAUSE_DATA` |
| **Deliverables** | Raw unstructured text + 2-sentence generic appeal | Color-coded audit table, 8th-grade patient summary, and formal GRO letter |

---

## 📂 Built-in Synthetic Preset Scenarios

DocuClaim includes 4 built-in 1-click presets covering critical FinTech and health insurance edge cases, plus 8 additional test scenarios in `data/synthetic_test_cases.json`:

1. **Preset 1: Room Rent Breach + Wrongful Pharmacy Deduction (Disputable)**
   - *Scenario*: Insured stayed in a room tariff of ₹10,000/day against a ₹5,000/day (1%) cap. Insurer settled room rent and doctor fees at 50%, but unlawfully applied a 50% cut to ₹25,000 in pharmacy expenses.
   - *Audit Verdict*: Room and doctor fee deductions are `VALID`. Pharmacy deduction of ₹12,500 is `INVALID` (exempt under Clause 4.1). Total Recoverable: **₹12,500**.
2. **Preset 2: Pre-Existing Disease (PED) 40-Month Tenure (Invalid Rejection)**
   - *Scenario*: Claim of ₹85,000 repudiated citing 36-month PED waiting period clause (Clause 7.2). Policy tenure is 40 continuous months.
   - *Audit Verdict*: Entire repudiation is `INVALID` as tenure (40 months) exceeds the waiting period (36 months). Total Recoverable: **₹85,000**.
3. **Preset 3: Valid ICU Sub-Limit Cap Deduction (Valid Deduction)**
   - *Scenario*: Sum Insured ₹4,00,000 with 2% daily ICU cap (₹8,000/day). Patient billed ₹12,000/day for 4 days. Insurer deducted ₹16,000.
   - *Audit Verdict*: ICU deduction of ₹16,000 is `VALID` under Clause 4.2. Total Recoverable: **₹0**.
4. **Preset 4: Missing Air Ambulance Clause Edge Case (Insufficient Data Guardrail)**
   - *Scenario*: Bill contains an Air Ambulance evacuation charge of ₹1,50,000. Provided policy wording does not contain an aeromedical clause.
   - *Audit Verdict*: Flags `INSUFFICIENT_CLAUSE_DATA` citing missing add-on policy schedule. Total Recoverable: **₹0 (Guardrail Triggered)**.

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Virtual environment (`venv`)

### 1. Clone & Navigate to Repository
```bash
git clone <repo-url>
cd Prompt_proejct
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API Key
You can pass your API key via environment variable:
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```
*Note: If no API key is provided, DocuClaim automatically runs in **Smart Offline Adjudication Mode**, allowing full deterministic evaluation of all presets with zero setup delay.*

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
├── app.py                              # Complete self-contained Streamlit application
├── requirements.txt                    # Project dependencies (streamlit, google-genai, etc.)
├── README.md                           # Comprehensive documentation & setup instructions
├── sample_policy.pdf                   # Test PDF policy wording for uploader verification
├── prompts/
│   └── prompt_iterations_v1_to_v4.md   # Evolutionary documentation of Prompts v1 to v4
└── data/
    └── synthetic_test_cases.json       # All 12 synthetic test scenarios with ground truth
```

---

## 👥 Team PromptCraft Credits

- **Team ID**: `PE-B13-57`
- **Batch Group**: `B13-24`
- **Course**: `SCSE4041 Prompt Engineering`
- **Team Members**:
  - **Aagam Jain** (`S24CSEU0263`)
  - **Riya Tyagi** (`S24CSEU0280`)
  - **Shubhashni Kant** (`S24CSEU0535`)
- **Developed for**: GenAI Health Insurance Triage & Dispute Resolution
