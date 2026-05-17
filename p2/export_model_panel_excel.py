from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_TITLE = "500 Ticker Model Metrics Panel"
DEFAULT_SHEET_NAME = "500 Ticker Panel"
DEFAULT_ROW_LABEL = "Metric"


def column_letter(index: int) -> str:
    result = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result.append(chr(65 + remainder))
    return "".join(reversed(result))


def load_csv(csv_path: Path) -> list[list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    return rows


def infer_widths(rows: list[list[str]], title: str) -> list[float]:
    column_count = max(len(row) for row in rows)
    widths = [12.0] * column_count
    widths[0] = max(widths[0], min(max(len(title), 18) * 1.1, 36.0))
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = min(max(widths[idx], len(value) * 1.1 + 2), 28.0)
    widths[0] = max(widths[0], 26.0)
    return widths


def make_cell(ref: str, value: str, style_id: int, *, is_number: bool) -> str:
    if is_number:
        return f'<c r="{ref}" s="{style_id}"><v>{value}</v></c>'
    return (
        f'<c r="{ref}" s="{style_id}" t="inlineStr">'
        f"<is><t>{escape(value)}</t></is></c>"
    )


def build_sheet_xml(rows: list[list[str]], widths: list[float], title: str, row_label: str) -> str:
    header = rows[0]
    data_rows = rows[1:]
    max_col = len(header)
    last_col = column_letter(max_col)
    xml_rows: list[str] = []

    xml_rows.append(
        "<row r=\"1\" ht=\"24\" customHeight=\"1\">"
      + make_cell("A1", title, 1, is_number=False)
        + "</row>"
    )

    header_cells = []
    for col_idx, value in enumerate([row_label, *header[1:]], start=1):
        ref = f"{column_letter(col_idx)}2"
        header_cells.append(make_cell(ref, value, 2, is_number=False))
    xml_rows.append("<row r=\"2\">" + "".join(header_cells) + "</row>")

    for row_offset, row in enumerate(data_rows, start=3):
        cells = []
        metric_name = row[0]
        cells.append(make_cell(f"A{row_offset}", metric_name, 3, is_number=False))
        for col_idx, raw_value in enumerate(row[1:], start=2):
            ref = f"{column_letter(col_idx)}{row_offset}"
            try:
                numeric = float(raw_value)
            except ValueError:
                cells.append(make_cell(ref, raw_value, 4, is_number=False))
                continue

            if numeric.is_integer():
                cells.append(make_cell(ref, str(int(numeric)), 4, is_number=True))
            else:
                cells.append(make_cell(ref, f"{numeric:.15g}", 5, is_number=True))
        xml_rows.append(f'<row r="{row_offset}">' + "".join(cells) + "</row>")

    cols_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{width:.2f}" customWidth="1"/>'
        for idx, width in enumerate(widths, start=1)
    )

    dimension_end = f"{last_col}{len(data_rows) + 2}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:{dimension_end}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="2" topLeftCell="A3" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols>{cols_xml}</cols>
  <sheetData>{''.join(xml_rows)}</sheetData>
  <mergeCells count="1"><mergeCell ref="A1:{last_col}1"/></mergeCells>
</worksheet>'''


def build_styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1">
    <numFmt numFmtId="164" formatCode="0.000000"/>
  </numFmts>
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="12"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color rgb="FF1F1F1F"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEFEFEF"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color auto="1"/></left>
      <right style="thin"><color auto="1"/></right>
      <top style="thin"><color auto="1"/></top>
      <bottom style="thin"><color auto="1"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="6">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="4" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def build_workbook_xml(sheet_name: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''


def build_app_xml(sheet_name: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python</Application>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr"><vt:lpstr>{escape(sheet_name)}</vt:lpstr></vt:vector>
  </TitlesOfParts>
</Properties>'''


def build_core_xml(title: str) -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>GitHub Copilot</dc:creator>
  <cp:lastModifiedBy>GitHub Copilot</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>'''


def write_workbook(
    csv_path: Path,
    output_path: Path,
    *,
    title: str,
    sheet_name: str,
    row_label: str,
) -> None:
    rows = load_csv(csv_path)
    widths = infer_widths(rows, title)
    sheet_xml = build_sheet_xml(rows, widths, title, row_label)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>''',
        )
        archive.writestr(
            "_rels/.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''',
        )
        archive.writestr("docProps/app.xml", build_app_xml(sheet_name))
        archive.writestr("docProps/core.xml", build_core_xml(title))
        archive.writestr("xl/workbook.xml", build_workbook_xml(sheet_name))
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''',
        )
        archive.writestr("xl/styles.xml", build_styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "csv_path",
    nargs="?",
    default="/Users/raj/ws/asset-pricing/p2/backtests/2026-05-16_23-33-18/reports/model_comparison_panel.csv",
  )
  parser.add_argument(
    "output_path",
    nargs="?",
    default="/Users/raj/ws/asset-pricing/p2/backtests/2026-05-16_23-33-18/reports/500_ticker_model_metrics_panel.xlsx",
  )
  parser.add_argument("--title", default=DEFAULT_TITLE)
  parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
  parser.add_argument("--row-label", default=DEFAULT_ROW_LABEL)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  csv_path = Path(args.csv_path)
  output_path = Path(args.output_path)

  write_workbook(
    csv_path,
    output_path,
    title=args.title,
    sheet_name=args.sheet_name,
    row_label=args.row_label,
  )
  print(output_path)
  return 0


if __name__ == "__main__":
    raise SystemExit(main())