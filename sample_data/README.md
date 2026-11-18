# sample_data

This is a shared, self-written corpus about **Nimbus Robotics**, a fictional company. Every RAG folder reads it through `DATA_DIR=../sample_data`. Facts deliberately span several files, so multi-hop and graph questions can only be answered by combining sources.

To regenerate it, run this from the repo root: `uv run --no-project --with python-docx,reportlab,openpyxl,pillow python scripts/make_sample_data.py`

| File | Content |
|------|---------|
| `text/company_overview.md` | Founders, products, locations, headcount |
| `text/release_notes.txt` | NimbusOS firmware 3.4.0–3.5.0 changes (3.4.2 fixes INC-2041) |
| `web/faq.html` | Warranty, operating temperature, support hours, update rollout |
| `docs/employee_handbook.docx` | PTO, remote work, on-call stipend, security, learning budget |
| `docs/incident_postmortem.docx` | INC-2041 Rotterdam fleet halt (author: X1 product lead) |
| `pdfs/q3_report.pdf` | Q3 2026 revenue, fleet size, largest customer |
| `pdfs/product_spec_nimbus_x1.pdf` | X1 payload, speed, battery, sensors, dimensions |
| `tables/pricing.xlsx` | List prices for hardware, software, warranty |
| `tables/support_tickets.csv` | Six support tickets with customer, firmware, status |
| `images/org_chart.png` (+ `.txt` caption) | Leadership reporting lines |
| `images/x1_diagram.png` (+ `.txt` caption) | Labeled X1 hardware diagram |
| `eval/questions.jsonl` | 20 eval questions (`modality`: text/table/image, `hops`: 1/2, one unanswerable) |

`.txt` files next to an image are captions. Text-only RAGs index the caption; Multimodal RAG uses the image itself.
