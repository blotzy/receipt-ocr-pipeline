"""
PDF generation and reporting functionality.
"""

import csv
import calendar
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

from .utils import money_fmt


def write_csv(rows: List[Dict], out_csv: Path):
    """Write receipts to CSV file."""
    fieldnames = ["date", "vendor", "amount", "category", "matcher", "notes", "source_file", "sha1"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def build_summary_pdf(rows: List[Dict], out_pdf: Path,
                      receipt_page_map: Optional[Dict[str, int]] = None,
                      title: str = "Loss of Use Receipts Summary",
                      use_links: bool = False) -> List[Dict]:
    """
    Build summary PDF with monthly grouping and clickable links to receipts.

    Args:
        rows: List of receipt data
        out_pdf: Output PDF path
        receipt_page_map: Dict mapping source_file to page number in final PDF
        title: Report title
        use_links: Whether to add clickable links (only works in final merged PDF)

    Returns:
        List of link metadata dicts with keys: page, x, y, width, height, target_page
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import blue

    receipt_page_map = receipt_page_map or {}
    use_links = use_links and len(receipt_page_map) > 0

    # Track link locations for annotation
    link_metadata = []

    # Group by month and compute totals
    monthly_data = defaultdict(list)
    monthly_totals = defaultdict(float)
    category_totals = defaultdict(float)

    for r in rows:
        date_str = r.get("date") or ""
        cat = r.get("category") or "Uncategorized"
        amt = r.get("amount") or 0.0

        # Extract year-month
        if date_str and len(date_str) >= 7:
            year_month = date_str[:7]  # YYYY-MM
        else:
            year_month = "Unknown"

        monthly_data[year_month].append(r)
        monthly_totals[year_month] += float(amt)
        category_totals[cat] += float(amt)

    c = canvas.Canvas(out_pdf.as_posix(), pagesize=letter)
    width, height = letter
    current_page_num = 0  # Track current page number for link annotations

    # Title page
    y = height - 1 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, title)
    y -= 0.3 * inch
    c.setFont("Helvetica", 10)
    timestamp = dt.datetime.now().isoformat(timespec='seconds')
    c.drawString(1 * inch, y, f"Generated: {timestamp}")
    y -= 0.4 * inch

    # Category Totals
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Category Totals")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    for cat, amt in sorted(category_totals.items()):
        c.drawString(1.1 * inch, y, f"{cat}: {money_fmt(amt)}")
        y -= 0.2 * inch
        if y < 1.2 * inch:
            c.showPage()
            current_page_num += 1
            y = height - 1 * inch

    # Monthly breakdown
    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Monthly Totals")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)

    for year_month in sorted(monthly_data.keys()):
        if year_month != "Unknown" and len(year_month) == 7:
            year, month = year_month.split("-")
            month_name = calendar.month_name[int(month)]
            display = f"{month_name} {year}"
        else:
            display = year_month

        c.drawString(1.1 * inch, y, f"{display}: {money_fmt(monthly_totals[year_month])}")
        y -= 0.2 * inch
        if y < 1.2 * inch:
            c.showPage()
            current_page_num += 1
            y = height - 1 * inch

    # Line items grouped by month
    for year_month in sorted(monthly_data.keys()):
        month_rows = sorted(monthly_data[year_month], key=lambda x: (x.get("date") or "", x.get("vendor") or ""))

        # Month header
        c.showPage()
        current_page_num += 1
        y = height - 1 * inch
        c.setFont("Helvetica-Bold", 14)

        if year_month != "Unknown" and len(year_month) == 7:
            year, month = year_month.split("-")
            month_name = calendar.month_name[int(month)]
            display = f"{month_name} {year}"
        else:
            display = year_month

        c.drawString(1 * inch, y, display)
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, y - 0.2 * inch, f"Total: {money_fmt(monthly_totals[year_month])}")
        y -= 0.5 * inch

        # Column headers
        c.setFont("Helvetica-Bold", 9)
        c.drawString(1.00 * inch, y, "Date")
        c.drawString(2.10 * inch, y, "Vendor")
        c.drawString(4.30 * inch, y, "Category")
        c.drawRightString(7.50 * inch, y, "Amount")
        y -= 0.15 * inch
        c.line(1.0 * inch, y, 7.6 * inch, y)
        y -= 0.15 * inch

        # Line items with clickable links
        c.setFont("Helvetica", 9)
        for r in month_rows:
            d = r.get("date") or ""
            v = (r.get("vendor") or "")[:28]
            cat = (r.get("category") or "")[:18]
            amt = money_fmt(r.get("amount"))
            source_file = r.get("source_file") or ""

            # Draw text
            c.drawString(1.00 * inch, y, d)

            # Draw vendor (color blue if it will link in final PDF)
            vendor_x = 2.10 * inch
            if use_links and source_file in receipt_page_map:
                c.setFillColor(blue)
                c.drawString(vendor_x, y, v)
                c.setFillColor("black")

                # Record link metadata for annotation
                # Estimate text width (9pt Helvetica ≈ 0.5 * len(text) points)
                text_width = len(v) * 4.5  # 9pt font, roughly 0.5pt per char
                link_metadata.append({
                    "page": current_page_num,
                    "x": vendor_x,
                    "y": y - 2,  # Slight offset for better clickable area
                    "width": text_width,
                    "height": 11,  # Font height + padding
                    "target_page": receipt_page_map[source_file]
                })
            else:
                c.drawString(vendor_x, y, v)

            c.drawString(4.30 * inch, y, cat)
            c.drawRightString(7.50 * inch, y, amt)
            y -= 0.18 * inch

            if y < 0.8 * inch:
                c.showPage()
                current_page_num += 1
                y = height - 1 * inch
                c.setFont("Helvetica-Bold", 12)
                c.drawString(1 * inch, y, f"{display} (cont.)")
                y -= 0.3 * inch
                c.setFont("Helvetica", 9)

    c.showPage()
    c.save()

    # Return link metadata for annotation
    return link_metadata


def add_pdf_link_annotations(pdf_path: Path, link_metadata: List[Dict]) -> None:
    """
    Add clickable link annotations to a PDF.

    Args:
        pdf_path: Path to PDF file (will be modified in place)
        link_metadata: List of dicts with keys: page, x, y, width, height, target_page
    """
    from pypdf import PdfWriter, PdfReader
    from pypdf.generic import (RectangleObject, DictionaryObject, NameObject,
                               NumberObject, ArrayObject)

    reader = PdfReader(pdf_path.as_posix())
    writer = PdfWriter()

    # Copy all pages
    for page in reader.pages:
        writer.add_page(page)

    # Add link annotations
    for link in link_metadata:
        page_num = link["page"]
        if page_num >= len(writer.pages):
            continue

        page = writer.pages[page_num]
        page_height = float(page.mediabox.height)

        # Create link annotation
        # PDF coordinates: (0,0) is bottom-left, ReportLab uses same
        x = link["x"]
        y = link["y"]
        width = link["width"]
        height = link["height"]
        target_page = link["target_page"]

        # Create rectangle for clickable area
        rect = RectangleObject([x, y, x + width, y + height])

        # Create link annotation dictionary
        link_dict = DictionaryObject()
        link_dict.update({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Link"),
            NameObject("/Rect"): rect,
            NameObject("/Border"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(0)]),  # No border
            NameObject("/Dest"): ArrayObject([
                writer.pages[target_page].indirect_reference,
                NameObject("/Fit")
            ])
        })

        # Add annotation to page
        if "/Annots" in page:
            page[NameObject("/Annots")].append(writer._add_object(link_dict))
        else:
            page[NameObject("/Annots")] = ArrayObject([writer._add_object(link_dict)])

    # Write back to same file
    with pdf_path.open("wb") as f:
        writer.write(f)


def merge_pdfs(summary_pdf: Path, receipt_files: List[Path],
               out_pdf: Path, rows: List[Dict]) -> Dict[str, int]:
    """
    Merge summary and receipts into final PDF, tracking page numbers.

    Args:
        summary_pdf: Summary PDF path
        receipt_files: List of receipt PDF paths
        out_pdf: Output merged PDF path
        rows: Receipt data to map source files to PDF files

    Returns:
        Dict mapping source_file to page number in final PDF
    """
    from pypdf import PdfWriter, PdfReader

    writer = PdfWriter()

    # Add summary pages
    summary_reader = PdfReader(summary_pdf.as_posix())
    summary_page_count = len(summary_reader.pages)
    writer.append(summary_reader)

    # Add receipt pages and track page numbers
    receipt_page_map = {}
    current_page = summary_page_count

    # Sort receipt files to match the order in rows (by date, vendor)
    sorted_rows = sorted(rows, key=lambda x: (x.get("date") or "", x.get("vendor") or ""))

    for row in sorted_rows:
        source_file = row.get("source_file", "")
        # Find matching receipt file
        matching_files = [f for f in receipt_files if f.name == source_file or f.stem + ".pdf" == source_file]

        if matching_files:
            receipt_file = matching_files[0]
            try:
                reader = PdfReader(receipt_file.as_posix())
                receipt_page_map[source_file] = current_page
                writer.append(reader)
                current_page += len(reader.pages)
            except Exception as e:
                print(f"[WARN] Skipping {receipt_file.name}: {e}")

    with out_pdf.open("wb") as f:
        writer.write(f)

    return receipt_page_map
