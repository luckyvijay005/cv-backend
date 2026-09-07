"""
Export utilities for classroom analytics reports
Generates PDF and CSV exports from analysis results
"""

import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors


def generate_pdf_report(report_data, output_path):
    """
    Generate professional PDF report from analysis results

    Args:
        report_data: Dictionary containing analysis report
        output_path: Path to save PDF file

    Returns:
        Path to generated PDF
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []

    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        alignment=1  # CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=10,
        spaceBefore=10
    )

    # Title
    title = Paragraph("Classroom Analytics Report", title_style)
    elements.append(title)

    # Report metadata
    date_str = report_data.get('date', 'Unknown')
    duration_min = report_data.get('duration_seconds', 0) / 60
    meta_text = f"<b>Generated:</b> {date_str} | <b>Duration:</b> {duration_min:.1f} minutes | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    elements.append(Paragraph(meta_text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # Summary section
    elements.append(Paragraph("Summary Statistics", heading_style))

    summary_data = [
        ['Metric', 'Value'],
        ['Total Students Detected', str(report_data.get('total_students', 0))],
        ['Video Duration', f"{duration_min:.1f} min"],
        ['Total Frames Analyzed', str(report_data.get('total_frames', 0))],
        ['Proximity Threshold',
            f"{report_data.get('proximity_threshold_px', 80)}px (≈1.5m)"],
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))

    # Top 3 least interactive students
    elements.append(
        Paragraph("Top 3 Least Interactive Students", heading_style))

    top_3 = report_data.get('top_3_least_interactive', [])
    if top_3:
        top3_data = [
            ['Rank', 'Student ID', 'Interaction Score',
                'Isolated (min)', 'With Peers (min)', 'Confidence'],
        ]

        for idx, student in enumerate(top_3):
            confidence_pct = f"{student.get('confidence', 0) * 100:.0f}%"
            top3_data.append([
                str(idx + 1),
                str(student.get('track_id', 'N/A')),
                f"{student.get('score', 0):.3f}",
                f"{student.get('isolated_minutes', 0):.2f}",
                f"{student.get('near_peer_minutes', 0):.2f}",
                confidence_pct,
            ])

        top3_table = Table(top3_data, colWidths=[
                           0.6*inch, 0.8*inch, 1.2*inch, 1.0*inch, 1.0*inch, 0.8*inch])
        top3_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        elements.append(top3_table)
    else:
        elements.append(Paragraph("No students detected.", styles['Normal']))

    elements.append(Spacer(1, 0.3*inch))

    # All students breakdown (if space allows)
    all_metrics = report_data.get('all_metrics', {})
    if all_metrics:
        elements.append(PageBreak())
        elements.append(Paragraph("All Students Breakdown", heading_style))

        # Sort by score
        sorted_students = sorted(
            all_metrics.items(),
            key=lambda x: x[1].get('score', 0)
        )

        all_data = [
            ['Student ID', 'Score',
                'Isolated (min)', 'With Peers (min)', 'Approach Events', 'Confidence'],
        ]

        for student_id, metrics in sorted_students:
            confidence_pct = f"{metrics.get('confidence', 0) * 100:.0f}%"
            all_data.append([
                str(student_id),
                f"{metrics.get('score', 0):.3f}",
                f"{metrics.get('isolated_minutes', 0):.2f}",
                f"{metrics.get('near_peer_minutes', 0):.2f}",
                str(metrics.get('approach_events', 0)),
                confidence_pct,
            ])

        all_table = Table(all_data, colWidths=[
                          1.0*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 0.8*inch])
        all_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        elements.append(all_table)

    # Build PDF
    doc.build(elements)
    return output_path


def generate_csv_report(report_data, output_path):
    """
    Generate CSV export with all student metrics

    Args:
        report_data: Dictionary containing analysis report
        output_path: Path to save CSV file

    Returns:
        Path to generated CSV
    """
    all_metrics = report_data.get('all_metrics', {})

    # Build dataframe
    rows = []
    for student_id, metrics in all_metrics.items():
        rows.append({
            'student_id': student_id,
            'interaction_score': metrics.get('score', 0),
            'isolated_minutes': metrics.get('isolated_minutes', 0),
            'near_peer_minutes': metrics.get('near_peer_minutes', 0),
            'approach_events': metrics.get('approach_events', 0),
            'confidence': metrics.get('confidence', 0),
        })

    df = pd.DataFrame(rows)

    # Sort by score (ascending = least interactive first)
    df = df.sort_values('interaction_score')

    # Save to CSV
    df.to_csv(output_path, index=False)

    return output_path
