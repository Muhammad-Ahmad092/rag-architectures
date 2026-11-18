"""Generate the shared ``sample_data/`` corpus (fictional "Nimbus Robotics" universe).

All content is self-written. Facts deliberately span files so multi-hop and
graph-style questions are answerable only by combining sources.

Run from the repo root:
    uv run --no-project --with python-docx,reportlab,openpyxl,pillow python scripts/make_sample_data.py
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "sample_data"
FIXED_DATE = dt.datetime(2026, 11, 1, 12, 0, 0)

# --------------------------------------------------------------------------- text


COMPANY_OVERVIEW = """# Nimbus Robotics — Company Overview

Nimbus Robotics builds autonomous robots for warehouse logistics. The company was
founded in 2019 in Portland, Oregon by **Amara Okafor** (CEO) and **Lukas Brandt** (CTO).

## Mission
Make every warehouse shift safer and faster by automating repetitive picking and sorting work.

## Products
- **Nimbus X1**: an autonomous mobile picking robot for aisles and shelving. Product lead: Priya Raman.
- **Nimbus S2**: a stationary sorting arm that places parcels into outbound chutes.
- **NimbusOS**: the fleet-management software that runs on every robot and in the cloud console.

## Locations
Headquarters are in Portland, Oregon. A European service hub opened in Rotterdam,
Netherlands in 2024 to support customers such as Halvorsen Logistics.

## People
Roughly 310 employees as of October 2026. Diego Martins leads Customer Support
and reports directly to the CEO.
"""

RELEASE_NOTES = """NimbusOS firmware release notes
===============================

3.5.0 (2026-09-30)
- New: dynamic aisle re-routing when an obstacle blocks a pick path for more than 20 seconds.
- New: battery health telemetry in the cloud console.
- Changed: default maximum speed of the Nimbus X1 lowered to 2.0 m/s in mixed human zones.

3.4.2 (2026-07-16)
- Fix: clock-synchronization bug that halted X1 fleets after a leap-second style NTP correction.
  This fix resolves incident INC-2041.
- Fix: S2 gripper calibration drift after 10,000 cycles.

3.4.1 (2026-06-20)
- New: faster fleet map loading (about 40% faster on sites with more than 200 robots).
- Known issue: rare clock drift under certain NTP configurations (see 3.4.2).

3.4.0 (2026-05-02)
- New: multi-floor support for Nimbus X1.
- New: role-based access control in the cloud console.
"""

FAQ_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Nimbus Robotics — FAQ</title></head>
<body>
  <nav>Home | Products | Support</nav>
  <h1>Frequently asked questions</h1>
  <h2>What is the warranty on Nimbus robots?</h2>
  <p>All Nimbus X1 and S2 robots include a 3-year limited hardware warranty. Batteries are covered for 2 years.</p>
  <h2>What temperatures can the Nimbus X1 operate in?</h2>
  <p>The X1 operates between -10 &deg;C and 45 &deg;C, which makes it suitable for chilled warehouses but not deep-freeze storage.</p>
  <h2>When is customer support available?</h2>
  <p>Standard support runs Monday to Friday, 07:00&ndash;19:00 Pacific Time. Enterprise customers get 24/7 phone support.</p>
  <h2>How do firmware updates work?</h2>
  <p>NimbusOS updates are staged: 5% of a fleet first, then the rest after 24 hours without critical alerts.</p>
  <footer>&copy; 2026 Nimbus Robotics (fictional company)</footer>
</body>
</html>
"""

# --------------------------------------------------------------------------- docx

HANDBOOK = [
    ("h", "Nimbus Robotics Employee Handbook (2026 edition)"),
    ("h2", "Time off"),
    ("p", "Full-time employees receive 25 days of paid time off per year, plus local public holidays."),
    ("p", "Unused paid time off up to 5 days may be carried over into the next calendar year."),
    ("h2", "Remote work"),
    ("p", "Employees may work remotely up to 3 days per week. Field service engineers are exempt and follow site schedules."),
    ("h2", "On-call"),
    ("p", "Support engineers join a weekly on-call rotation. On-call engineers receive a stipend of $400 per week."),
    ("h2", "Security"),
    ("p", "Badges must be worn at all times on customer sites. Lost badges must be reported to Security within 2 hours."),
    ("h2", "Learning budget"),
    ("p", "Each employee has an annual learning budget of $2,000 for courses, books and conferences."),
]

POSTMORTEM = [
    ("h", "Postmortem: INC-2041 — X1 fleet halt at Rotterdam"),
    ("p", "Author: Priya Raman (Product Lead, Nimbus X1). Date of incident: 2026-07-14."),
    ("h2", "Summary"),
    ("p", "The entire Nimbus X1 fleet at the Halvorsen Logistics warehouse in Rotterdam stopped moving for "
          "4 hours and 12 minutes. Robots entered a safe-stop state after their clocks disagreed with the site "
          "controller by more than 500 ms."),
    ("h2", "Root cause"),
    ("p", "Firmware 3.4.1 handled a large NTP time correction incorrectly, so robot clocks drifted apart."),
    ("h2", "Resolution"),
    ("p", "Engineers rolled the fleet back to 3.4.0 as a mitigation. The permanent fix shipped in firmware 3.4.2."),
    ("h2", "Impact"),
    ("p", "About 18,000 order lines were delayed. Halvorsen Logistics received a service credit of EUR 25,000."),
    ("h2", "Action items"),
    ("p", "Add clock-skew alerting to the cloud console; add NTP fault injection to the release test suite."),
]

# --------------------------------------------------------------------------- pdf

Q3_REPORT = [
    ("Nimbus Robotics — Q3 2026 Quarterly Report", "title"),
    ("Highlights", "h"),
    ("Revenue for Q3 2026 was $18.4 million, up 31% year over year.", "p"),
    ("Recurring NimbusOS subscription revenue reached $4.1 million in the quarter.", "p"),
    ("Deployed fleet: 1,240 Nimbus X1 units and 385 Nimbus S2 units across 46 customer sites.", "p"),
    ("Customers", "h"),
    ("Our largest customer by deployed units is Halvorsen Logistics (Rotterdam), with 160 X1 robots.", "p"),
    ("Net revenue retention was 124%.", "p"),
    ("Outlook", "h"),
    ("We expect Q4 2026 revenue between $20 million and $21 million, driven by the S2 holiday season.", "p"),
]

X1_SPEC = [
    ("Nimbus X1 — Product Specification Sheet", "title"),
    ("Product lead: Priya Raman", "p"),
    ("Performance", "h"),
    ("Payload capacity: 25 kg. Maximum speed: 2.2 m/s (2.0 m/s in mixed human zones since firmware 3.5.0).", "p"),
    ("Battery: 10 hours of runtime per charge; full recharge takes 90 minutes.", "p"),
    ("Sensing", "h"),
    ("One 360-degree LiDAR on the top mast and four depth cameras, one on each side.", "p"),
    ("Physical", "h"),
    ("Footprint 80 cm x 60 cm, height 1.9 m, weight 140 kg.", "p"),
    ("Operating temperature: -10 C to 45 C.", "p"),
]

# --------------------------------------------------------------------------- tables

PRICING = [
    ("product", "type", "list_price_usd", "billing"),
    ("Nimbus X1", "hardware", 48000, "one-time"),
    ("Nimbus S2", "hardware", 32500, "one-time"),
    ("NimbusOS Standard", "software", 299, "per robot per month"),
    ("NimbusOS Enterprise", "software", 449, "per robot per month"),
    ("Extended warranty (+2 years)", "service", 6000, "per robot, one-time"),
]

TICKETS = [
    ("ticket_id", "customer", "product", "firmware", "issue", "status", "opened"),
    ("T-1001", "Halvorsen Logistics", "Nimbus X1", "3.4.1", "Fleet safe-stop, clock skew", "resolved", "2026-07-14"),
    ("T-1002", "Brightline Grocers", "Nimbus S2", "3.4.1", "Gripper calibration drift", "resolved", "2026-07-02"),
    ("T-1003", "Halvorsen Logistics", "Nimbus X1", "3.4.2", "Slow map loading on floor 2", "resolved", "2026-08-09"),
    ("T-1004", "Kestrel Pharma", "Nimbus X1", "3.5.0", "Battery health warning", "open", "2026-10-03"),
    ("T-1005", "Brightline Grocers", "Nimbus X1", "3.5.0", "Re-routing too aggressive", "open", "2026-10-11"),
    ("T-1006", "Kestrel Pharma", "NimbusOS", "3.5.0", "RBAC role missing in console", "in progress", "2026-10-20"),
]

# --------------------------------------------------------------------------- images

ORG_CAPTION = (
    "Org chart of Nimbus Robotics leadership. Amara Okafor (CEO) is at the top. "
    "Reporting to the CEO: Lukas Brandt (CTO) and Diego Martins (Head of Customer Support). "
    "Reporting to the CTO: Priya Raman (Product Lead, Nimbus X1) and Mei Tanaka (Product Lead, Nimbus S2)."
)
DIAGRAM_CAPTION = (
    "Labeled diagram of the Nimbus X1 robot. The 360-degree LiDAR sits on top of the mast. "
    "Four depth cameras are mounted one per side. The suction gripper is on the front arm. "
    "The swappable battery bay is at the rear, near the charging contacts at the bottom."
)


def write_text() -> None:
    (ROOT / "text").mkdir(parents=True, exist_ok=True)
    (ROOT / "web").mkdir(parents=True, exist_ok=True)
    (ROOT / "text" / "company_overview.md").write_text(COMPANY_OVERVIEW, encoding="utf-8")
    (ROOT / "text" / "release_notes.txt").write_text(RELEASE_NOTES, encoding="utf-8")
    (ROOT / "web" / "faq.html").write_text(FAQ_HTML, encoding="utf-8")


def write_docx() -> None:
    from docx import Document

    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    for name, blocks in (("employee_handbook.docx", HANDBOOK), ("incident_postmortem.docx", POSTMORTEM)):
        doc = Document()
        doc.core_properties.created = FIXED_DATE
        doc.core_properties.modified = FIXED_DATE
        doc.core_properties.author = "Nimbus Robotics (fictional)"
        for kind, text in blocks:
            if kind == "h":
                doc.add_heading(text, level=0)
            elif kind == "h2":
                doc.add_heading(text, level=1)
            else:
                doc.add_paragraph(text)
        doc.save(ROOT / "docs" / name)


def write_pdf() -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    style_map = {"title": styles["Title"], "h": styles["Heading2"], "p": styles["BodyText"]}
    (ROOT / "pdfs").mkdir(parents=True, exist_ok=True)
    for name, blocks in (("q3_report.pdf", Q3_REPORT), ("product_spec_nimbus_x1.pdf", X1_SPEC)):
        doc = SimpleDocTemplate(str(ROOT / "pdfs" / name), pagesize=A4, invariant=1)
        story = []
        for text, kind in blocks:
            story += [Paragraph(text, style_map[kind]), Spacer(1, 6)]
        doc.build(story)


def write_tables() -> None:
    from openpyxl import Workbook

    (ROOT / "tables").mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.properties.created = FIXED_DATE
    wb.properties.modified = FIXED_DATE
    ws = wb.active
    ws.title = "pricing"
    for row in PRICING:
        ws.append(row)
    wb.save(ROOT / "tables" / "pricing.xlsx")
    with open(ROOT / "tables" / "support_tickets.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(TICKETS)


def _box(draw, xy, text, font):
    draw.rectangle(xy, outline="black", width=2, fill="#eef3fb")
    x0, y0, x1, y1 = xy
    draw.multiline_text(((x0 + x1) / 2, (y0 + y1) / 2), text, fill="black", font=font, anchor="mm", align="center")


def write_images() -> None:
    from PIL import Image, ImageDraw, ImageFont

    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)

    img = Image.new("RGB", (900, 420), "white")
    d = ImageDraw.Draw(img)
    _box(d, (350, 20, 550, 90), "Amara Okafor\nCEO", font)
    _box(d, (150, 170, 350, 240), "Lukas Brandt\nCTO", font)
    _box(d, (550, 170, 790, 240), "Diego Martins\nHead of Support", font)
    _box(d, (30, 320, 270, 390), "Priya Raman\nProduct Lead X1", font)
    _box(d, (300, 320, 540, 390), "Mei Tanaka\nProduct Lead S2", font)
    for a, b in (((450, 90), (250, 170)), ((450, 90), (670, 170)), ((250, 240), (150, 320)), ((250, 240), (420, 320))):
        d.line([a, b], fill="black", width=2)
    img.save(ROOT / "images" / "org_chart.png", optimize=True)
    (ROOT / "images" / "org_chart.txt").write_text(ORG_CAPTION, encoding="utf-8")

    img = Image.new("RGB", (700, 520), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((250, 200, 450, 440), outline="black", width=3, fill="#dde7dd")  # body
    d.rectangle((335, 60, 365, 200), outline="black", width=3)  # mast
    d.ellipse((315, 30, 385, 80), outline="black", width=3, fill="#ffd27f")  # lidar
    d.rectangle((450, 260, 600, 285), outline="black", width=3)  # arm
    d.ellipse((595, 250, 640, 295), outline="black", width=3)  # gripper
    d.rectangle((180, 300, 250, 400), outline="black", width=3, fill="#f4c7c3")  # battery (rear)
    for cx, cy in ((250, 230), (450, 230), (300, 440), (400, 440)):
        d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill="black")  # cameras
    labels = [((400, 40), "360° LiDAR"), ((560, 225), "Suction gripper"), ((30, 340), "Battery bay (rear)"),
              ((460, 470), "Depth cameras x4"), ((375, 130), "Mast")]
    for xy, text in labels:
        d.text(xy, text, fill="black", font=font)
    img.save(ROOT / "images" / "x1_diagram.png", optimize=True)
    (ROOT / "images" / "x1_diagram.txt").write_text(DIAGRAM_CAPTION, encoding="utf-8")


def main() -> None:
    write_text()
    write_docx()
    write_pdf()
    write_tables()
    write_images()
    print(f"sample data written to {ROOT}")


if __name__ == "__main__":
    main()
