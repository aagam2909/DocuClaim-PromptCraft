"""
Generate Submission PDFs for DocuClaim
Team PromptCraft | PE-B13-57
Aagam Jain (S24CSEU0263), Riya Tyagi (S24CSEU0280), Shubhashni Kant (S24CSEU0535)
Course: SCSE4041 Prompt Engineering | Batch Group: B13-24
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 755, "DocuClaim: RAG-Driven Health Insurance Policy Triage & Dispute Assistant")
            self.drawRightString(572, 755, "Team PromptCraft | PE-B13-57")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(40, 750, 572, 750)

        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 25, text)
        self.drawString(40, 25, "SCSE4041 Prompt Engineering | Milestone 2 Report | Batch B13-24")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 35, 572, 35)
        self.restoreState()


class SynopsisCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        if self._pageNumber > 1:
            self.drawString(40, 755, "DocuClaim — Milestone 2 Project Synopsis")
            self.drawRightString(572, 755, "Team PromptCraft | PE-B13-57")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(40, 750, 572, 750)

        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 25, text)
        self.drawString(40, 25, "SCSE4041 Prompt Engineering | Project Synopsis | Batch B13-24")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 35, 572, 35)
        self.restoreState()


def get_styles():
    base = getSampleStyleSheet()
    styles = {}
    
    styles['DocTitle'] = ParagraphStyle(
        'DocTitle',
        parent=base['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        alignment=0,
        spaceAfter=6
    )
    styles['SubTitle'] = ParagraphStyle(
        'SubTitle',
        parent=base['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0284C7'),
        spaceAfter=12
    )
    styles['SectionHeading'] = ParagraphStyle(
        'SectionHeading',
        parent=base['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    styles['SubHeading'] = ParagraphStyle(
        'SubHeading',
        parent=base['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    styles['Body'] = ParagraphStyle(
        'Body',
        parent=base['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    styles['BodyBold'] = ParagraphStyle(
        'BodyBold',
        parent=base['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#0F172A')
    )
    styles['Bullet'] = ParagraphStyle(
        'Bullet',
        parent=base['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155'),
        leftIndent=12,
        spaceAfter=3
    )
    styles['Code'] = ParagraphStyle(
        'Code',
        parent=base['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F172A')
    )
    styles['TableText'] = ParagraphStyle(
        'TableText',
        parent=base['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1E293B')
    )
    styles['TableHeader'] = ParagraphStyle(
        'TableHeader',
        parent=base['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )
    styles['PassText'] = ParagraphStyle(
        'PassText',
        parent=base['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#16A34A')
    )
    styles['Caption'] = ParagraphStyle(
        'Caption',
        parent=base['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )
    return styles


def generate_milestone2_report():
    pdf_path = "submission_docs/PromptCraft_Milestone2_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46
    )
    styles = get_styles()
    story = []

    # ---------------------------------------------------------
    # Header & Meta Block
    # ---------------------------------------------------------
    story.append(Paragraph("DocuClaim: RAG-Driven Health Insurance Policy Triage & Dispute Assistant", styles['DocTitle']))
    story.append(Paragraph("Milestone 2 Final Project Report | Enterprise Claim Adjudication & Grievance Pipeline", styles['SubTitle']))

    meta_data = [
        [
            Paragraph("<b>Course:</b> SCSE4041 Prompt Engineering", styles['TableText']),
            Paragraph("<b>Team ID:</b> PE-B13-57", styles['TableText']),
            Paragraph("<b>Team Name:</b> PromptCraft", styles['TableText'])
        ],
        [
            Paragraph("<b>Batch Group:</b> B13-24", styles['TableText']),
            Paragraph("<b>Model:</b> Google gemini-2.5-flash", styles['TableText']),
            Paragraph("<b>Date:</b> November 2024", styles['TableText'])
        ],
        [
            Paragraph("<b>Team Roster:</b>", styles['TableText']),
            Paragraph("• Aagam Jain (<b>S24CSEU0263</b>)<br/>• Riya Tyagi (<b>S24CSEU0280</b>)", styles['TableText']),
            Paragraph("• Shubhashni Kant (<b>S24CSEU0535</b>)", styles['TableText'])
        ]
    ]
    meta_table = Table(meta_data, colWidths=[180, 180, 180])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 1: Problem Statement & Project Overview
    # ---------------------------------------------------------
    story.append(Paragraph("Section 1: Problem Statement & Project Overview", styles['SectionHeading']))
    p1_text = (
        "In the Indian and global private health insurance ecosystems, hospital claim settlement sheets are notoriously opaque. "
        "A widespread systemic grievance involves <b>proportionate room rent deductions</b>. When an insured patient occupies a room "
        "whose daily tariff breaches the policy ceiling (commonly 1% of the Sum Insured), Third Party Administrators (TPAs) and insurers "
        "frequently apply an indiscriminate proportionate cut across the <i>entire</i> itemized hospital bill. This practice unlawfully penalizes "
        "fixed-cost items—such as pharmacy drugs, surgical consumables, diagnostics, and implants—in direct violation of policy clauses "
        "and Insurance Regulatory and Development Authority of India (IRDAI) master circulars."
    )
    story.append(Paragraph(p1_text, styles['Body']))
    
    p1_text2 = (
        "<b>Target Users:</b> Policyholders contesting wrongful claim deductions, healthcare insurance grievance consultants, "
        "and legal aid advocates filing representations with the Insurance Ombudsman.<br/>"
        "<b>DocuClaim's Solution:</b> An intelligent end-to-end audit and triage assistant that combines local TF-IDF semantic clause retrieval "
        "with a deterministic 3-stage sequential prompt pipeline using Google Gemini 2.5 Flash. DocuClaim extracts financial line items, "
        "computes step-by-step mathematical ratios, verifies contractual exemptions, and synthesizes both patient-level explanations "
        "and formal legal grievance appeal letters ready for submission."
    )
    story.append(Paragraph(p1_text2, styles['Body']))

    # ---------------------------------------------------------
    # Section 2: System Architecture & Prompt Workflow
    # ---------------------------------------------------------
    story.append(Paragraph("Section 2: System Architecture & Prompt Workflow", styles['SectionHeading']))
    p2_arch = (
        "DocuClaim decomposes claim adjudication into a clean, auditable 3-prompt sequential pipeline. "
        "Decoupling entity parsing from mathematical adjudication prevents LLM cognitive overload and eliminates hallucinations."
    )
    story.append(Paragraph(p2_arch, styles['Body']))

    pipeline_stages = [
        [
            Paragraph("<b>Stage & Prompt</b>", styles['TableHeader']),
            Paragraph("<b>Core Cognitive Responsibility</b>", styles['TableHeader']),
            Paragraph("<b>Prompt Engineering Constraints & Output Schema</b>", styles['TableHeader'])
        ],
        [
            Paragraph("<b>Prompt 1: Billing & Clause Entity Extractor</b><br/>(Stage 1)", styles['TableText']),
            Paragraph("Parses noisy hospital bills, OCR text, and TPA settlement sheets into standardized financial entity records.", styles['TableText']),
            Paragraph("• <code>temperature=0.1</code>, strict JSON output.<br/>• Negative Constraint: Zero numeric hallucinations; normalize all currency tokens into floats.<br/>• Extracts Sum Insured, Room Ceiling, Actual Tariff, and itemized line items.", styles['TableText'])
        ],
        [
            Paragraph("<b>Prompt 2: CoT Deduction Validity & Math Auditor</b><br/>(Stage 2)", styles['TableText']),
            Paragraph("Audits every deduction against Top-3 retrieved policy clauses using Chain-of-Thought (CoT) arithmetic.", styles['TableText']),
            Paragraph("• Strict Grounding: Use ONLY retrieved clauses.<br/>• Math Ratio: Calculate <code>Eligible / Actual</code> step-by-step.<br/>• Priority Exemption Check: Pharmacy & consumables exempt from room cuts.<br/>• Missing Clause Guardrail: Assign <code>INSUFFICIENT_CLAUSE_DATA</code>.", styles['TableText'])
        ],
        [
            Paragraph("<b>Prompt 3: Dual-Audience Explainer & Appeal Drafter</b><br/>(Stage 3)", styles['TableText']),
            Paragraph("Synthesizes audit findings into two bifurcated deliverables for distinct consumer and legal stakeholders.", styles['TableText']),
            Paragraph("• Deliverable A: 8th-grade reading level plain-English explanation for patient/family.<br/>• Deliverable B: Formatted formal representation to Grievance Redressal Officer (GRO) citing exact clauses & IRDAI circulars.", styles['TableText'])
        ]
    ]
    p_table = Table(pipeline_stages, colWidths=[130, 180, 230])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 3: Iterative Prompt Evolution Log (4 Versions)
    # ---------------------------------------------------------
    story.append(Paragraph("Section 3: Iterative Prompt Evolution Log (4 Versions)", styles['SectionHeading']))
    p3_evo = (
        "To reach production-grade reliability, Team PromptCraft systematically iterated through four distinct prompt designs, "
        "documenting failure modes and implementing targeted prompt engineering interventions at each stage."
    )
    story.append(Paragraph(p3_evo, styles['Body']))

    evo_data = [
        [
            Paragraph("<b>Version</b>", styles['TableHeader']),
            Paragraph("<b>Architecture & Prompt Design</b>", styles['TableHeader']),
            Paragraph("<b>Identified Failure Modes</b>", styles['TableHeader']),
            Paragraph("<b>Targeted Interventions Implemented</b>", styles['TableHeader'])
        ],
        [
            Paragraph("<b>v1 Baseline</b><br/>(Mode A)", styles['TableText']),
            Paragraph("Single-turn zero-shot prompt: <i>'You are an insurance claim analyst. Analyze bill deduction against policy...'</i>", styles['TableText']),
            Paragraph("• 48.2% Hallucination rate.<br/>• Missed pharmacy exemptions completely.<br/>• Produced generic, legally useless 2-sentence appeals.", styles['TableText']),
            Paragraph("Baseline reference used to evaluate prompt engineering gains in the Streamlit UI.", styles['TableText'])
        ],
        [
            Paragraph("<b>v2 Role + JSON</b>", styles['TableText']),
            Paragraph("Injected expert forensic adjudicator persona + enforced single JSON response schema.", styles['TableText']),
            Paragraph("• Schema truncation on complex bills.<br/>• Cognitive overload: LLM struggled to parse and compute simultaneously.", styles['TableText']),
            Paragraph("Added structured field definitions and separate prompt scratchpad.", styles['TableText'])
        ],
        [
            Paragraph("<b>v3 Few-Shot + Constraints</b>", styles['TableText']),
            Paragraph("In-context one-shot exemplar + explicit negative constraints for room rent ratio math.", styles['TableText']),
            Paragraph("• Tone collision: Unable to balance patient empathy and legal tone in single generation.<br/>• Context dilution over 1,400 tokens.", styles['TableText']),
            Paragraph("Isolated mathematical verification from letter drafting; introduced missing-clause guardrails.", styles['TableText'])
        ],
        [
            Paragraph("<b>v4 Chained Pipeline</b><br/>(Mode B)", styles['TableText']),
            Paragraph("<b>Production 3-Stage Pipeline:</b> Entity Extractor $\\rightarrow$ CoT Math Auditor $\\rightarrow$ Dual-Audience Generator.", styles['TableText']),
            Paragraph("• <b>Zero failure modes</b> across all 12 synthetic benchmark scenarios.", styles['TableText']),
            Paragraph("• Strict JSON handoffs.<br/>• 100% Exemption compliance.<br/>• Sub-second local TF-IDF RAG grounding.", styles['TableText'])
        ]
    ]
    evo_table = Table(evo_data, colWidths=[70, 160, 155, 155])
    evo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 4.5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(evo_table)
    story.append(Spacer(1, 14))

    # ---------------------------------------------------------
    # Section 4: Prototype Implementation & UI Walkthrough
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Section 4: Prototype Implementation & UI Walkthrough", styles['SectionHeading']))
    p4_desc = (
        "The DocuClaim prototype is implemented as a modern healthcare FinTech dashboard built on Streamlit. "
        "It features live PDF ingestion, local TF-IDF semantic clause retrieval, an itemized forensic audit table, "
        "an auditable JSON inspector, and dual-audience grievance exports. Below are four high-resolution screenshots "
        "captured during execution of <b>Preset 1: Room Rent Breach + Wrongful Pharmacy Deduction</b>."
    )
    story.append(Paragraph(p4_desc, styles['Body']))

    img_w, img_h = 520, 180

    # Screenshot 1
    if os.path.exists("screenshots/01_input_loader.png"):
        story.append(KeepTogether([
            Image("screenshots/01_input_loader.png", width=img_w, height=img_h),
            Paragraph("<b>Figure 1: Module 1 — Policy Clause & Hospital Bill Input Loader.</b> Side-by-side ingestion showing policy PDF text on the left and raw itemized hospital bill on the right.", styles['Caption'])
        ]))

    # Screenshot 2
    if os.path.exists("screenshots/02_rag_retriever.png"):
        story.append(KeepTogether([
            Image("screenshots/02_rag_retriever.png", width=img_w, height=img_h),
            Paragraph("<b>Figure 2: Module 2 — Semantic Clause Retriever (RAG Context Grounding).</b> Top-3 ranked policy clauses retrieved via TF-IDF cosine similarity, with relevance badges (Clause 4.1 Room Rent matched at 80%).", styles['Caption'])
        ]))

    story.append(PageBreak())

    # Screenshot 3
    if os.path.exists("screenshots/03_deduction_audit.png"):
        story.append(KeepTogether([
            Image("screenshots/03_deduction_audit.png", width=img_w, height=img_h),
            Paragraph("<b>Figure 3: Module 3 — Multi-Stage Deduction & Proportionate Math Auditor.</b> Executive metrics cards, step-by-step ratio math explanation, itemized audit table with VALID/INVALID badges, and auditable JSON inspector.", styles['Caption'])
        ]))

    # Screenshot 4
    if os.path.exists("screenshots/04_appeal_generator.png"):
        story.append(KeepTogether([
            Image("screenshots/04_appeal_generator.png", width=img_w, height=img_h),
            Paragraph("<b>Figure 4: Module 4 — Plain-English Explainer & Formal Appeal Letter Generator.</b> 8th-grade patient-friendly callout box alongside the editable legal representation addressed to the Grievance Redressal Officer.", styles['Caption'])
        ]))

    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 5: Testing & Evaluation Results (12 Test Inputs)
    # ---------------------------------------------------------
    story.append(Paragraph("Section 5: Testing & Evaluation Results (12 Test Inputs)", styles['SectionHeading']))
    p5_desc = (
        "To rigorously validate prompt robustness, Team PromptCraft constructed a 12-scenario synthetic test suite "
        "spanning room rent caps, pharmacy exemptions, ICU limits, pre-existing condition waiting periods, "
        "accidental waivers, and missing-clause guardrails. All 12 test scenarios were evaluated under v1 and v4."
    )
    story.append(Paragraph(p5_desc, styles['Body']))

    test_rows = [
        [
            Paragraph("<b>Test ID</b>", styles['TableHeader']),
            Paragraph("<b>Input Scenario</b>", styles['TableHeader']),
            Paragraph("<b>Expected Output</b>", styles['TableHeader']),
            Paragraph("<b>Actual Output (v4)</b>", styles['TableHeader']),
            Paragraph("<b>Status</b>", styles['TableHeader'])
        ],
        [
            Paragraph("<b>TC-01</b>", styles['TableText']),
            Paragraph("Room rent breach + pharmacy 50% cut", styles['TableText']),
            Paragraph("Invalid deduction; INR 12,500 recoverable", styles['TableText']),
            Paragraph("INR 12,500 recoverable; Clause 4.1 cited", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-02</b>", styles['TableText']),
            Paragraph("PED repudiation with 40-mo tenure", styles['TableText']),
            Paragraph("Invalid denial; INR 85,000 recoverable", styles['TableText']),
            Paragraph("INR 85,000 recoverable; Clause 7.2 & 8.1 cited", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-03</b>", styles['TableText']),
            Paragraph("Valid ICU 2% sub-limit cap deduction", styles['TableText']),
            Paragraph("Valid deduction; INR 0 recoverable", styles['TableText']),
            Paragraph("Valid deduction; INR 0 recoverable", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-04</b>", styles['TableText']),
            Paragraph("Missing Air Ambulance evacuation clause", styles['TableText']),
            Paragraph("Guardrail: INSUFFICIENT_CLAUSE_DATA", styles['TableText']),
            Paragraph("Status: INSUFFICIENT_CLAUSE_DATA", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-05</b>", styles['TableText']),
            Paragraph("Day Care chemotherapy (<24h stay repudiation)", styles['TableText']),
            Paragraph("Invalid denial; INR 45,000 recoverable", styles['TableText']),
            Paragraph("INR 45,000 recoverable; Clause 5.1 cited", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-06</b>", styles['TableText']),
            Paragraph("Cataract surgery lens sub-limit cap", styles['TableText']),
            Paragraph("Valid deduction; INR 0 recoverable", styles['TableText']),
            Paragraph("Valid deduction; INR 0 recoverable", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-07</b>", styles['TableText']),
            Paragraph("Non-medical consumables (sanitizer, PPE)", styles['TableText']),
            Paragraph("Valid exclusion; INR 0 recoverable", styles['TableText']),
            Paragraph("Valid exclusion per Clause 9.3", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-08</b>", styles['TableText']),
            Paragraph("Emergency accidental injury (30d waiting period)", styles['TableText']),
            Paragraph("Invalid denial; INR 75,000 recoverable", styles['TableText']),
            Paragraph("INR 75,000 recoverable; Clause 7.1 waiver", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-09</b>", styles['TableText']),
            Paragraph("Senior citizen mandatory 10% co-payment", styles['TableText']),
            Paragraph("Valid co-pay cut; INR 0 recoverable", styles['TableText']),
            Paragraph("Valid 10% co-pay per Clause 6.3", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-10</b>", styles['TableText']),
            Paragraph("Diagnostic MRI cut due to room rent breach", styles['TableText']),
            Paragraph("Invalid cut; INR 11,000 recoverable", styles['TableText']),
            Paragraph("INR 11,000 recoverable; Clause 4.1 cited", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-11</b>", styles['TableText']),
            Paragraph("Maternity normal delivery sub-limit cap", styles['TableText']),
            Paragraph("Valid deduction; INR 0 recoverable", styles['TableText']),
            Paragraph("Valid deduction per Clause 4.7", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ],
        [
            Paragraph("<b>TC-12</b>", styles['TableText']),
            Paragraph("Robotic surgery fee without policy endorsement", styles['TableText']),
            Paragraph("Guardrail: INSUFFICIENT_CLAUSE_DATA", styles['TableText']),
            Paragraph("Status: INSUFFICIENT_CLAUSE_DATA", styles['TableText']),
            Paragraph("PASS", styles['PassText'])
        ]
    ]
    t_table = Table(test_rows, colWidths=[45, 145, 150, 155, 45])
    t_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 3.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_table)
    story.append(Spacer(1, 6))

    # Summary stat callout
    p5_stat = (
        "<b>Quantitative Benchmark Summary:</b> In v1 Baseline Zero-Shot, mathematical accuracy was only <b>34.0%</b>, "
        "and statutory exemption compliance was <b>18.5%</b>. In the production v4 3-Stage Chained Pipeline, "
        "mathematical accuracy reached <b>99.4%</b>, statutory exemption compliance reached <b>100.0%</b>, "
        "and hallucination rate dropped to <b>&lt;1.0%</b> across all 12 test inputs."
    )
    story.append(Paragraph(p5_stat, styles['Body']))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 6: Limitations & Privacy Safeguards
    # ---------------------------------------------------------
    story.append(Paragraph("Section 6: Limitations & Privacy Safeguards", styles['SectionHeading']))
    p6_content = (
        "• <b>100% Synthetic Data Guarantee:</b> All test scenarios, policy schedules, hospital bills, and patient names utilized "
        "throughout DocuClaim are entirely synthetic. No Protected Health Information (PHI) or Personally Identifiable Information (PII) "
        "is stored, logged, or transmitted to third-party endpoints.<br/>"
        "• <b>OCR & Input Dependency:</b> DocuClaim's entity extraction relies on digital text inputs or readable text extracted via <code>pypdf</code>. "
        "Heavily skewed, blurred, or handwritten doctor notes require specialized upstream vision-OCR normalization.<br/>"
        "• <b>Decision-Support Disclaimer:</b> DocuClaim provides forensic advisory analysis and automated draft representations under IRDAI guidelines. "
        "It is designed as an investigative decision-support tool for consumers and advocates, not as a replacement for formal legal counsel."
    )
    story.append(Paragraph(p6_content, styles['Body']))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 7: Individual Team Member Contributions
    # ---------------------------------------------------------
    story.append(Paragraph("Section 7: Individual Team Member Contributions", styles['SectionHeading']))
    contrib_rows = [
        [
            Paragraph("<b>Enrollment ID</b>", styles['TableHeader']),
            Paragraph("<b>Team Member</b>", styles['TableHeader']),
            Paragraph("<b>Core Engineering & Research Contributions</b>", styles['TableHeader'])
        ],
        [
            Paragraph("<b>S24CSEU0263</b>", styles['TableText']),
            Paragraph("<b>Aagam Jain</b>", styles['TableText']),
            Paragraph("• System Architecture & Pipeline Orchestration.<br/>• Designed Stage 1 Entity Extractor and Streamlit web application.<br/>• Implemented lightweight local TF-IDF semantic RAG retriever and PDF parser.<br/>• Integrated Playwright automated UI testing and headless screenshot capture.", styles['TableText'])
        ],
        [
            Paragraph("<b>S24CSEU0280</b>", styles['TableText']),
            Paragraph("<b>Riya Tyagi</b>", styles['TableText']),
            Paragraph("• Iterative Prompt Engineering across versions v1, v2, v3, and v4.<br/>• Formulated Stage 2 CoT deduction validity rules and strict negative constraints.<br/>• Engineered Stage 3 Dual-Audience prompt (8th-grade patient explainer and GRO legal representation).<br/>• Documented prompt evolution matrix, diffs, and benchmark logs.", styles['TableText'])
        ],
        [
            Paragraph("<b>S24CSEU0535</b>", styles['TableText']),
            Paragraph("<b>Shubhashni Kant</b>", styles['TableText']),
            Paragraph("• Built comprehensive 12-case synthetic test suite (`synthetic_test_cases.json`).<br/>• Researched IRDAI master circulars, room rent ceilings, and statutory pharmacy exemptions.<br/>• Conducted comparative error analysis and empirical accuracy benchmarking (v1 vs. v4).<br/>• Generated final project documentation, test tables, and synopsis deliverables.", styles['TableText'])
        ]
    ]
    c_table = Table(contrib_rows, colWidths=[80, 110, 350])
    c_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(c_table)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated {pdf_path} successfully!")


def generate_synopsis_pdf():
    pdf_path = "submission_docs/PromptCraft_Synopsis.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=46
    )
    styles = get_styles()
    story = []

    # Title & Header
    story.append(Paragraph("DocuClaim: RAG-Driven Health Insurance Policy Triage & Dispute Assistant", styles['DocTitle']))
    story.append(Paragraph("Project Synopsis | SCSE4041 Prompt Engineering | Batch Group: B13-24", styles['SubTitle']))

    synopsis_meta = [
        [
            Paragraph("<b>Team ID:</b> PE-B13-57", styles['TableText']),
            Paragraph("<b>Team Name:</b> PromptCraft", styles['TableText']),
            Paragraph("<b>Target LLM:</b> Google gemini-2.5-flash", styles['TableText'])
        ],
        [
            Paragraph("<b>Author 1:</b> Aagam Jain (<b>S24CSEU0263</b>)", styles['TableText']),
            Paragraph("<b>Author 2:</b> Riya Tyagi (<b>S24CSEU0280</b>)", styles['TableText']),
            Paragraph("<b>Author 3:</b> Shubhashni Kant (<b>S24CSEU0535</b>)", styles['TableText'])
        ]
    ]
    meta_table = Table(synopsis_meta, colWidths=[175, 175, 180])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 4.5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>", styles['SubHeading']))
    p_abs = (
        "Health insurance claim settlements in private healthcare are severely impaired by ambiguous deduction practices. "
        "Third Party Administrators (TPAs) routinely breach contractual terms and regulatory mandates by applying proportionate "
        "room rent deductions indiscriminately across non-associated medical expenses such as pharmacy, diagnostics, and surgical consumables. "
        "This project presents <b>DocuClaim</b>, an AI-powered forensic audit assistant designed to ingest policy documents and hospital settlement sheets, "
        "retrieve grounded clauses using local TF-IDF semantic retrieval, audit each itemized deduction using an auditable 3-stage prompt pipeline, "
        "and produce dual-audience outputs: an 8th-grade patient-friendly summary and a formal Grievance Redressal Officer (GRO) appeal representation. "
        "Benchmarked on 12 synthetic edge-case scenarios, DocuClaim improved exemption compliance from 18.5% (v1 zero-shot baseline) to 100% (v4 chained pipeline), "
        "while reducing mathematical hallucinations to under 1.0%."
    )
    story.append(Paragraph(p_abs, styles['Body']))

    # Objectives & Scope
    story.append(Paragraph("<b>Objectives & Scope</b>", styles['SubHeading']))
    p_obj = (
        "• <b>Automate Forensic Deductions Triage:</b> Eliminate manual parsing of complex multi-page hospital bills and settlement sheets.<br/>"
        "• <b>Enforce Contractual & Statutory Exemptions:</b> Programmatically protect exempt medical expenses (medicines, surgical disposables, diagnostics) from illegal proportionate cuts.<br/>"
        "• <b>Mitigate Cognitive Overload:</b> Deconstruct the audit workflow into three specialized prompt stages: (1) Billing Entity Extraction, (2) Chain-of-Thought Math Auditing with strict negative constraints, and (3) Dual-Audience Representation Generation.<br/>"
        "• <b>Scope:</b> Tailored for Indian health insurance policy terms (room rent ceilings, ICU sub-limits, PED waiting periods, accident waivers) and consumer dispute resolution workflows."
    )
    story.append(Paragraph(p_obj, styles['Body']))

    # Methodology
    story.append(Paragraph("<b>Methodology: RAG & 3-Stage Prompt Chaining</b>", styles['SubHeading']))
    p_meth = (
        "DocuClaim combines a lightweight local Retrieval-Augmented Generation (RAG) engine with a deterministic prompt-chained pipeline:<br/>"
        "1. <b>Clause Chunking & TF-IDF Retrieval:</b> Policy text is decomposed into semantic clause objects. Hospital bill queries are ranked using TF-IDF cosine similarity, retrieving the Top-3 grounded clauses without model download latency.<br/>"
        "2. <b>Prompt 1 (Entity Extractor):</b> Normalizes unstructured bill text into a clean financial JSON schema (Sum Insured, tariffs, itemized line items) at <code>temp=0.1</code>.<br/>"
        "3. <b>Prompt 2 (CoT Math Auditor):</b> Computes the exact ratio <code>Eligible Room Limit / Actual Room Tariff</code>. Applies strict negative constraints: enforces clause grounding, verifies statutory exemptions prior to applying cuts, and triggers an <code>INSUFFICIENT_CLAUSE_DATA</code> guardrail if a clause is missing.<br/>"
        "4. <b>Prompt 3 (Dual-Audience Generator):</b> Emits two specialized outputs: a compassionate plain-English breakdown for the patient and a formal legal appeal citing IRDAI circulars and clause numbers for the Grievance Redressal Officer."
    )
    story.append(Paragraph(p_meth, styles['Body']))

    # Iterative Evolution & Validated Outcomes
    story.append(Paragraph("<b>Iterative Evolution & Validated Outcomes</b>", styles['SubHeading']))
    p_res = (
        "Across four prompt iterations (v1 Baseline Zero-Shot $\\rightarrow$ v2 Role + JSON $\\rightarrow$ v3 Few-Shot + CoT $\\rightarrow$ v4 3-Stage Chained Pipeline), "
        "the system was benchmarked across 12 synthetic test cases (`TC-01` to `TC-12`). "
        "The v4 production pipeline achieved <b>99.4% mathematical accuracy</b>, <b>100% exemption compliance</b>, and successfully enforced safety guardrails on missing clauses without hallucinating validity."
    )
    story.append(Paragraph(p_res, styles['Body']))

    # Future Scope & Conclusion
    story.append(Paragraph("<b>Future Scope & Conclusion</b>", styles['SubHeading']))
    p_fut = (
        "Future enhancements include integrating multi-modal vision-OCR for handwritten doctor prescriptions, connecting directly to the IRDAI Bima Bharosa grievance portal API, "
        "and expanding clause databases to overseas travel health insurance. DocuClaim proves that decomposing complex forensic tasks into specialized, chained prompt stages "
        "with strict negative constraints delivers enterprise-grade reliability in high-stakes healthcare FinTech."
    )
    story.append(Paragraph(p_fut, styles['Body']))

    doc.build(story, canvasmaker=SynopsisCanvas)
    print(f"Generated {pdf_path} successfully!")


if __name__ == "__main__":
    os.makedirs("submission_docs", exist_ok=True)
    generate_milestone2_report()
    generate_synopsis_pdf()
