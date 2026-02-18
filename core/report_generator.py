"""Report generation for screening results — PDF, JSON, and plain text."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.utils import AnalysisResult, ProgressCallback


class ReportGenerator:
    """Generates exportable reports from screening results."""

    def generate_pdf(
        self,
        result: AnalysisResult,
        output_path: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> bool:
        """Generate a PDF report with findings, heatmap, and disclaimer."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                Image,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

            if on_progress:
                on_progress(1, 4, "Creating PDF layout...")

            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                leftMargin=20 * mm,
                rightMargin=20 * mm,
                topMargin=20 * mm,
                bottomMargin=20 * mm,
            )

            styles = getSampleStyleSheet()
            elements = []

            # Title
            title_style = ParagraphStyle(
                "ReportTitle",
                parent=styles["Title"],
                fontSize=20,
                spaceAfter=6,
            )
            elements.append(Paragraph("LocalMedScan Screening Report", title_style))
            elements.append(Spacer(1, 4 * mm))

            # Disclaimer banner
            disclaimer_style = ParagraphStyle(
                "Disclaimer",
                parent=styles["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#92400E"),
                backColor=colors.HexColor("#FEF3C7"),
                borderColor=colors.HexColor("#F59E0B"),
                borderWidth=1,
                borderPadding=8,
                spaceBefore=4,
                spaceAfter=8,
            )
            elements.append(Paragraph(
                f"<b>WARNING:</b> {result.disclaimer}", disclaimer_style
            ))
            elements.append(Spacer(1, 4 * mm))

            if on_progress:
                on_progress(2, 4, "Adding findings...")

            # Metadata
            meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
            screening_type = result.screening_type.value.upper()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elements.append(Paragraph(f"Screening Type: {screening_type}", meta_style))
            elements.append(Paragraph(f"Model: {result.model_name}", meta_style))
            elements.append(Paragraph(f"Processing Time: {result.processing_time_ms}ms", meta_style))
            elements.append(Paragraph(f"Date: {timestamp}", meta_style))
            elements.append(Spacer(1, 6 * mm))

            # Findings table
            if result.findings:
                elements.append(Paragraph("Findings", styles["Heading2"]))

                table_data = [["Condition", "Confidence", "Severity"]]
                for finding in result.findings:
                    pct = f"{finding.confidence * 100:.1f}%"
                    table_data.append([finding.name, pct, finding.severity.value.upper()])

                table = Table(table_data, colWidths=[200, 80, 80])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 6 * mm))
            else:
                elements.append(Paragraph("No significant findings detected.", styles["Normal"]))
                elements.append(Spacer(1, 6 * mm))

            if on_progress:
                on_progress(3, 4, "Adding images...")

            # Heatmap image
            if result.heatmap_path and Path(result.heatmap_path).exists():
                elements.append(Paragraph("Attention Heatmap", styles["Heading2"]))
                elements.append(Paragraph(
                    "Areas highlighted by the AI model during analysis. "
                    "Brighter regions indicate higher attention.",
                    meta_style,
                ))
                elements.append(Spacer(1, 2 * mm))
                img = Image(result.heatmap_path, width=160, height=160)
                elements.append(img)
                elements.append(Spacer(1, 6 * mm))

            # Full disclaimer at bottom
            elements.append(Spacer(1, 10 * mm))
            full_disclaimer_style = ParagraphStyle(
                "FullDisclaimer",
                parent=styles["Normal"],
                fontSize=8,
                textColor=colors.HexColor("#6B7280"),
                spaceBefore=8,
            )
            from i18n import t
            elements.append(Paragraph(t("disclaimer.full"), full_disclaimer_style))

            # Footer
            footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
            elements.append(Spacer(1, 4 * mm))
            elements.append(Paragraph(
                "Generated by LocalMedScan | Svetozar Technologies | "
                "AI Should Be Free. AI Should Be Private. AI Should Be Yours.",
                footer_style,
            ))

            if on_progress:
                on_progress(4, 4, "Writing PDF...")

            doc.build(elements)
            return True

        except Exception:
            return False

    def generate_json(self, result: AnalysisResult, output_path: str) -> bool:
        """Generate a JSON export of the screening results."""
        try:
            data = {
                "tool": "LocalMedScan",
                "version": "1.0.0",
                "organization": "Svetozar Technologies",
                "timestamp": datetime.now().isoformat(),
                "disclaimer": result.disclaimer,
                "screening_type": result.screening_type.value,
                "model_name": result.model_name,
                "input_path": result.input_path,
                "processing_time_ms": result.processing_time_ms,
                "overall_confidence": result.overall_confidence,
                "findings": [
                    {
                        "name": f.name,
                        "confidence": f.confidence,
                        "severity": f.severity.value,
                        "description": f.description,
                    }
                    for f in result.findings
                ],
                "heatmap_path": result.heatmap_path,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True

        except Exception:
            return False

    def generate_txt(self, result: AnalysisResult, output_path: str) -> bool:
        """Generate a plain text report."""
        try:
            lines = [
                "=" * 60,
                "LOCALMEDSCAN SCREENING REPORT",
                "Svetozar Technologies",
                "=" * 60,
                "",
                f"WARNING: {result.disclaimer}",
                "",
                f"Screening Type: {result.screening_type.value.upper()}",
                f"Model: {result.model_name}",
                f"Processing Time: {result.processing_time_ms}ms",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "-" * 40,
                "FINDINGS",
                "-" * 40,
            ]

            if result.findings:
                for finding in result.findings:
                    pct = f"{finding.confidence * 100:.1f}%"
                    lines.append(f"  {finding.name}: {pct} ({finding.severity.value.upper()})")
                    if finding.description:
                        lines.append(f"    {finding.description}")
                    lines.append("")
            else:
                lines.append("  No significant findings detected.")
                lines.append("")

            lines.extend([
                "-" * 40,
                "DISCLAIMER",
                "-" * 40,
                result.disclaimer,
                "",
                "Generated by LocalMedScan | Svetozar Technologies",
                "AI Should Be Free. AI Should Be Private. AI Should Be Yours.",
            ])

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return True

        except Exception:
            return False
