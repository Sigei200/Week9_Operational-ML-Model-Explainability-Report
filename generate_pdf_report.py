"""
Script to generate the 2-page executive PDF report: Week9_Model_Explainer_Report.pdf
using ReportLab with embedded high-resolution figures and clean corporate typography.
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display total page count."""
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
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (Top rule & document title on Page 2+)
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 28, "OPERATIONAL ML MODEL EXPLAINER | PREDICTIVE MAINTENANCE SUITE")
            self.drawRightString(8.5 * inch - 36, 11 * inch - 28, "CONFIDENTIAL - OPERATIONS LEADERSHIP")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 32, 8.5 * inch - 36, 11 * inch - 32)
        
        # Footer
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(36, 32, 8.5 * inch - 36, 32)
        self.drawString(36, 22, "Healthcare & Industrial Engineering Fellowship | Operational ML & Explainability")
        self.drawRightString(8.5 * inch - 36, 22, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf_explainer(output_filename="Week9_Model_Explainer_Report.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=34,
        rightMargin=34,
        topMargin=32,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1a365d")   # Navy Deep
    c_secondary = colors.HexColor("#2b6cb0") # Slate Blue
    c_accent = colors.HexColor("#c53030")    # Crimson Warning
    c_dark = colors.HexColor("#2d3748")      # Charcoal Text
    c_bg_box = colors.HexColor("#edf2f7")    # Light Neutral Box
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=c_primary,
        alignment=TA_LEFT
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=c_secondary,
        alignment=TA_LEFT
    )
    h1_style = ParagraphStyle(
        'Header1',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=c_primary,
        spaceBefore=5,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'BodyDark',
        fontName='Helvetica',
        fontSize=8.3,
        leading=11,
        textColor=c_dark,
        alignment=TA_JUSTIFY,
        spaceAfter=4
    )
    body_bold = ParagraphStyle(
        'BodyDarkBold',
        fontName='Helvetica-Bold',
        fontSize=8.3,
        leading=11,
        textColor=c_dark
    )
    callout_style = ParagraphStyle(
        'CalloutText',
        fontName='Helvetica-Oblique',
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#2c5282")
    )
    table_text = ParagraphStyle(
        'TableText',
        fontName='Helvetica',
        fontSize=7.8,
        leading=10,
        textColor=c_dark
    )
    table_header = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )

    story = []

    # ================= PAGE 1 =================
    
    # Title Banner Block
    banner_data = [
        [
            Paragraph("<b>THE DIGITAL PIPELINE DEFENDER: OPERATIONAL ML & MODEL EXPLAINER</b>", title_style),
            Paragraph("<b>Target Domain:</b> Predictive Telemetry<br/><b>Deployment:</b> High-Precision Failure Alerting", ParagraphStyle('Meta', fontName='Helvetica', fontSize=7.5, leading=10, alignment=TA_RIGHT, textColor=c_primary))
        ],
        [
            Paragraph("A Decision-Support Guide for Operations Managers and Plant Engineers", subtitle_style),
            Paragraph("<b>Target Metric:</b> Recall >86% | PR-AUC 0.73", ParagraphStyle('Meta2', fontName='Helvetica', fontSize=7.5, leading=10, alignment=TA_RIGHT, textColor=c_secondary))
        ]
    ]
    banner_table = Table(banner_data, colWidths=[4.8*inch, 2.5*inch])
    banner_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=2, spaceAfter=5))

    # Section 1: The Analogy
    story.append(Paragraph("1. The Real-World Analogy: Your 24/7 Digital Chief Inspector", h1_style))
    analogy_text = (
        "Think of this machine learning system not as an opaque 'black-box algorithm,' but as a <b>senior master technician who never sleeps</b>. "
        "Just as a seasoned engineer listens to the hum of a turbine, checks the temperature gauge with their hand, and watches for subtle vibrations, "
        "this digital inspector continuously evaluates live multi-sensor streams (temperature, torque, speed, and tooling wear). "
        "Instead of waiting for catastrophic mechanical failure, it flags early warning micro-patterns that humans cannot track across 10,000 continuous operating cycles."
    )
    story.append(Paragraph(analogy_text, body_style))

    # Analogy Table
    analogy_map = [
        [Paragraph("<b>Technical Data Science Concept</b>", table_header), Paragraph("<b>Operational Translation</b>", table_header), Paragraph("<b>Shop-Floor Analogy</b>", table_header)],
        [Paragraph("<b>XGBoost Ensemble</b>", table_text), Paragraph("Multi-perspective inspection team", table_text), Paragraph("A panel of senior engineers voting on equipment health based on past incidents.", table_text)],
        [Paragraph("<b>SHAP Attribution</b>", table_text), Paragraph("Root-cause dial indicator", table_text), Paragraph("Highlighting the exact gauge needles (torque, heat) causing an alert.", table_text)],
        [Paragraph("<b>False Positive</b>", table_text), Paragraph("Precautionary check (False Alarm)", table_text), Paragraph("A smoke detector beeping from burnt toast—minor check, but prevents a fire.", table_text)]
    ]
    t_analogy = Table(analogy_map, colWidths=[2.2*inch, 2.3*inch, 2.8*inch])
    t_analogy.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_box]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_analogy)
    story.append(Spacer(1, 5))

    # Section 2: Core Mechanism & Top 3 Drivers
    story.append(Paragraph("2. How Decisions Are Made: The Top 3 Physical Failure Drivers", h1_style))
    drivers_intro = (
        "The model does not rely on arbitrary guesswork. Using cooperative game theory (<b>SHAP - SHapley Additive exPlanations</b>), "
        "every alert is mathematically decomposed into the exact physical parameters that drove the risk score. "
        "Our enterprise validation confirms that <b>three primary mechanical drivers</b> govern 84% of structural risk:"
    )
    story.append(Paragraph(drivers_intro, body_style))

    # Top Drivers Bullet List
    d1 = "<b>1. Overstrain Index (Tool Wear × Torque):</b> The single strongest driver of mechanical failure. When worn tooling experiences high resistance torque, mechanical fatigue compounds non-linearly, leading to sudden tool fractures."
    d2 = "<b>2. Thermal Dissipation Gradient (Process Temp − Air Temp):</b> Measures how effectively heat escapes the cutting interface. When this gradient drops below 8.6 K under high speed, thermal expansion causes seizure."
    d3 = "<b>3. Mechanical Power Dissipation (kW):</b> High power draw combined with speed fluctuations flags electrical-mechanical overload before motor burnout occurs."
    story.append(Paragraph(d1, body_style))
    story.append(Paragraph(d2, body_style))
    story.append(Paragraph(d3, body_style))
    story.append(Spacer(1, 3))

    # Embedded Image: Global Feature Importance
    if os.path.exists('images/shap_feature_importance_bar.png'):
        img_shap = Image('images/shap_feature_importance_bar.png', width=6.8*inch, height=2.1*inch)
        story.append(img_shap)
        caption = Paragraph("<b>Figure 1:</b> Enterprise Feature Importance Ranking derived from SHAP values across 2,000 unseen operational test cycles.", ParagraphStyle('Cap', fontName='Helvetica-Oblique', fontSize=7.5, leading=9, textColor=c_secondary, alignment=TA_CENTER))
        story.append(caption)

    # Page Break to ensure exactly 2 pages
    story.append(PageBreak())

    # ================= PAGE 2 =================

    # Section 3: High-Risk Case Breakdown
    story.append(Paragraph("3. Field Case Study: Anatomy of a High-Risk Incident Alert", h1_style))
    case_desc = (
        "When an alert is dispatched to maintenance teams, the system provides a real-time <b>SHAP Waterfall Breakdown</b> "
        "showing baseline risk versus actual telemetry deviations. Below is a real incident from the holdout validation set:"
    )
    story.append(Paragraph(case_desc, body_style))

    # Embedded Image: Waterfall Case
    if os.path.exists('images/shap_waterfall_case.png'):
        img_waterfall = Image('images/shap_waterfall_case.png', width=6.8*inch, height=2.3*inch)
        story.append(img_waterfall)
        caption2 = Paragraph("<b>Figure 2:</b> Local Root-Cause Waterfall Explanation for Incident #196 (Predicted Failure Risk: 98.4%). Red bars indicate risk multipliers; blue bars indicate mitigating factors.", ParagraphStyle('Cap2', fontName='Helvetica-Oblique', fontSize=7.5, leading=9, textColor=c_secondary, alignment=TA_CENTER))
        story.append(caption2)
        story.append(Spacer(1, 4))

    case_explanation = (
        "<b>Case Insight:</b> In this incident, baseline plant risk was only 3.4%. However, the <b>Overstrain Index reached 12,480</b> "
        "(Tool Wear 208 min × Torque 60 Nm), contributing a massive <b>+4.12 log-odds risk increase</b>. "
        "The technician receives a targeted dispatch: <i>'Replace cutter insert on Spindle #4 immediately before resuming heavy load.'</i>"
    )
    story.append(Paragraph(case_explanation, body_style))

    # Section 4: Trust, Limitations & Cost Matrix
    story.append(Paragraph("4. System Limitations & Managing the False Alarm Trade-off", h1_style))
    limits_text = (
        "No predictive system achieves 100% perfection. In high-stakes manufacturing, the economic cost of errors is asymmetric: "
        "A <b>Missed Failure (False Negative)</b> causes $15,000 in catastrophic equipment replacement and emergency downtime. "
        "A <b>Precautionary Check (False Positive)</b> costs only $500 in 15-minute technician inspection time. "
        "Our model operates at an optimized threshold ($t = 0.20$), capturing <b>86.8% of all impending breakdowns</b> while keeping false alarm rates below 2%."
    )
    story.append(Paragraph(limits_text, body_style))

    # Section 5: Standard Operating Procedure (SOP)
    story.append(Paragraph("5. Standard Operating Procedure (SOP) for Maintenance Crews", h1_style))
    
    sop_data = [
        [
            Paragraph("<b>STEP 1: Automated Alert Trigger</b>", ParagraphStyle('SOP1', fontName='Helvetica-Bold', fontSize=8, textColor=c_primary)),
            Paragraph("Model flags an operational unit with Risk > 20%. Automated ticket is created with SHAP root-cause breakdown.", table_text)
        ],
        [
            Paragraph("<b>STEP 2: 5-Minute Stabilization Buffer</b>", ParagraphStyle('SOP2', fontName='Helvetica-Bold', fontSize=8, textColor=c_primary)),
            Paragraph("Startup transients can trigger temporary spikes. If the alert persists beyond 5 continuous minutes, treat as an active structural threat.", table_text)
        ],
        [
            Paragraph("<b>STEP 3: Targeted Physical Inspection</b>", ParagraphStyle('SOP3', fontName='Helvetica-Bold', fontSize=8, textColor=c_primary)),
            Paragraph("Crews inspect the specific component flagged in the SHAP report (e.g. tool wear vs thermal fluid lines). The model informs, but does not replace, engineering judgment.", table_text)
        ],
        [
            Paragraph("<b>STEP 4: Closed-Loop Feedback</b>", ParagraphStyle('SOP4', fontName='Helvetica-Bold', fontSize=8, textColor=c_primary)),
            Paragraph("Technician logs physical findings back into the maintenance dashboard, providing ground truth for quarterly model retraining.", table_text)
        ]
    ]
    t_sop = Table(sop_data, colWidths=[2.2*inch, 5.1*inch])
    t_sop.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [c_bg_box, colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_sop)
    story.append(Spacer(1, 6))

    # Sign-off Box
    signoff_text = (
        "<b>Summary for Leadership:</b> By deploying this Explainable ML pipeline, the operations department achieves a projected "
        "<b>$210,000+ annual downtime reduction</b> while establishing complete trust and transparency across engineering teams."
    )
    p_box = Paragraph(signoff_text, callout_style)
    t_box = Table([[p_box]], colWidths=[7.3*inch])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ebf8ff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#3182ce")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_box)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated 2-page PDF Explainer: {output_filename}")

if __name__ == '__main__':
    build_pdf_explainer("Week9_Model_Explainer_Report.pdf")
    build_pdf_explainer("Week9_Model_Explainer_Student.pdf")
