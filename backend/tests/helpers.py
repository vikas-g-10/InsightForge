"""Small synthetic fixture builders. No real data or personal information."""

import io
import zipfile

from openpyxl import Workbook

URL = "/api/ingest/validate"
SENTINEL = "ZZ-SENTINEL-REVIEW-TEXT-ZZ"

BASIC_CSV = (
    "review_text,rating,date\n"
    "Great battery life,5,2024-01-01\n"
    "Terrible fit and finish,1,2024-01-02\n"
)


def xlsx_bytes(rows, extra_sheet=False) -> bytes:
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    if extra_sheet:
        wb.create_sheet("Other").append(["ignored"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def upload(client, name, data, content_type="application/octet-stream", **form):
    return client.post(URL, files={"file": (name, data, content_type)}, data=form)


def error_of(response):
    return response.json()["error"]


def zip_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()
