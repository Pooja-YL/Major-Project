from datetime import datetime
from io import BytesIO

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAVY = colors.HexColor("#0B1F33")
CYAN = colors.HexColor("#10B7C8")
INK = colors.HexColor("#12263A")
MUTED = colors.HexColor("#62788A")
LINE = colors.HexColor("#DDE8F0")
PALE = colors.HexColor("#F2F7FA")
WHITE = colors.white


def _make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="Eyebrow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=CYAN,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=NAVY,
        spaceBefore=9,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="BodyReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=INK,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="SmallMuted",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="HeaderDetails",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=11,
        textColor=WHITE,
    ))
    styles.add(ParagraphStyle(
        name="TableLabel",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="TableValue",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=INK,
    ))
    return styles


def _image_flowable(source, max_width, max_height):
    if source is None:
        return Paragraph("Not available for this result.", _make_styles()["SmallMuted"])

    if isinstance(source, PILImage.Image):
        image_bytes = BytesIO()
        source.convert("RGB").save(image_bytes, format="PNG")
        image_bytes.seek(0)
        source = image_bytes

    with PILImage.open(source) as image_file:
        width, height = image_file.size
    scale = min(max_width / width, max_height / height)
    return Image(source, width=width * scale, height=height * scale)


def _section_heading(text, styles):
    return Paragraph(text, styles["SectionTitle"])


def _metric_table(rows, widths, header=False):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), PALE))
    table.setStyle(TableStyle(style))
    return table


def _draw_page(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(15 * mm, 14 * mm, A4[0] - 15 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(15 * mm, 9.5 * mm, "BoneSight  |  AI-assisted analysis report")
    canvas.drawRightString(A4[0] - 15 * mm, 9.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def generate_medical_report(output_path, report_data):
    styles = _make_styles()
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=19 * mm,
        title="BoneSight Analysis Report",
        author="BoneSight",
    )
    story = []
    timestamp = report_data["generated_at"]
    report_id = report_data["report_id"]

    title_block = Table(
        [[
            Paragraph("BONESIGHT", styles["Eyebrow"]),
            Paragraph("AI ANALYSIS REPORT", styles["SmallMuted"]),
        ], [
            Paragraph("X-ray study summary", styles["ReportTitle"]),
            Paragraph(f"<b>Report ID</b><br/>{report_id}<br/><br/><b>Prediction ID</b><br/>{report_data['prediction_id']}<br/><br/><b>Generated</b><br/>{timestamp}", styles["HeaderDetails"]),
        ]],
        colWidths=[112 * mm, 68 * mm],
    )
    title_block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([title_block, Spacer(1, 5 * mm)])

    story.append(_section_heading("Classification summary", styles))
    summary_rows = [
        [Paragraph("BODY PART", styles["TableLabel"]), Paragraph("CONFIDENCE", styles["TableLabel"]), Paragraph("FRACTURE STATUS", styles["TableLabel"]), Paragraph("FRACTURE CONFIDENCE", styles["TableLabel"])],
        [Paragraph(str(report_data["body_part"]), styles["TableValue"]), Paragraph(report_data["body_part_confidence"], styles["TableValue"]), Paragraph(str(report_data["fracture_status"]), styles["TableValue"]), Paragraph(report_data["fracture_confidence"], styles["TableValue"])],
        [Paragraph("FRACTURE TYPE", styles["TableLabel"]), Paragraph("TYPE CONFIDENCE", styles["TableLabel"]), Paragraph("STUDY FILE", styles["TableLabel"]), Paragraph("REPORT TIME", styles["TableLabel"])],
        [Paragraph(str(report_data["fracture_type"]), styles["TableValue"]), Paragraph(report_data["fracture_type_confidence"], styles["TableValue"]), Paragraph(str(report_data["study_name"]), styles["BodyReport"]), Paragraph(timestamp, styles["BodyReport"])],
    ]
    story.append(_metric_table(summary_rows, [45 * mm] * 4, header=True))

    quality = report_data["image_quality"]
    story.append(_section_heading("Image quality analysis", styles))
    quality_rows = [
        [Paragraph("BRIGHTNESS", styles["TableLabel"]), Paragraph("CONTRAST", styles["TableLabel"]), Paragraph("SHARPNESS", styles["TableLabel"]), Paragraph("NOISE ESTIMATE", styles["TableLabel"])],
        [Paragraph(f"{quality['brightness']:.1f}", styles["TableValue"]), Paragraph(f"{quality['contrast']:.1f}", styles["TableValue"]), Paragraph(f"{quality['sharpness']:.1f}", styles["TableValue"]), Paragraph(f"{quality['noise']:.1f}", styles["TableValue"])],
    ]
    story.append(_metric_table(quality_rows, [45 * mm] * 4, header=True))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"Resolution: {quality['width']} x {quality['height']} pixels", styles["SmallMuted"]))

    story.append(_section_heading("X-ray images", styles))
    image_width = 84 * mm
    image_height = 61 * mm
    source_images = [
        [Paragraph("UPLOADED X-RAY", styles["TableLabel"]), Paragraph("ENHANCED X-RAY", styles["TableLabel"])],
        [_image_flowable(report_data["original_image"], image_width, image_height), _image_flowable(report_data["enhanced_image"], image_width, image_height)],
    ]
    image_table = Table(source_images, colWidths=[90 * mm, 90 * mm], hAlign="LEFT")
    image_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(image_table)

    story.append(_section_heading("AI analysis summary", styles))
    story.append(Paragraph(report_data["summary"], styles["BodyReport"]))

    story.append(PageBreak())
    story.append(Paragraph("VISUAL REVIEW", styles["Eyebrow"]))
    story.append(Paragraph("Model-generated visualizations", styles["SectionTitle"]))
    story.append(Paragraph("These views accompany the recorded model outputs for this study.", styles["BodyReport"]))
    story.append(Spacer(1, 5 * mm))
    visualization_rows = [
        [Paragraph("GRAD-CAM", styles["TableLabel"]), Paragraph("FRACTURE LOCALIZATION", styles["TableLabel"])],
        [
            _image_flowable(report_data["gradcam_image"], image_width, 105 * mm),
            _image_flowable(report_data["localization_image"], image_width, 105 * mm),
        ],
    ]
    visualization_table = Table(visualization_rows, colWidths=[90 * mm, 90 * mm], hAlign="LEFT")
    visualization_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 1), (-1, 1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(visualization_table)

    document.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)
    return str(output_path)