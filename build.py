"""Download and index full annual reports. Runtime files live in data/."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = json.loads((ROOT / "reports.json").read_text())
SECTION = re.compile(r"第\s*[一二三四五六七八九十]+\s*节\s*[\u3400-\u9fff、及\s]{2,45}$")
FINANCIAL_LABEL = re.compile(
    r"营业(?:总)?收入|总资产|经营活动产生的现金流量净额|基本每股收益|"
    r"加权平均净资产收益率|归属于.{0,20}股东的净(?:资产|利润)"
)
FINANCIAL_NUMBER = re.compile(r"[-+−]?\d[\d,]*\.\d+%?")
ANNUAL_COLUMN = re.compile(r"20\d{2}\s*年(?:度|末)?|本(?:年|期).{0,18}增减[^\s]*")
WORD = re.compile(r"[\u3400-\u9fff]+|[a-z]+|\d+(?:\.\d+)?")
DIMENSIONS = 3072


def normalize(text):
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFKC", text)).strip()


def tokens(text):
    for word in WORD.findall(normalize(text).lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            yield from word
            yield from (word[i:i+2] for i in range(len(word)-1))
        else:
            yield word


def download(report):
    pdf = DATA / "pdf" / (report["code"] + ".pdf")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if not pdf.exists():
        temporary = pdf.with_suffix(".part")
        subprocess.run(["curl", "-fLsS", "--retry", "2", "--max-time", "120",
                        report["url"], "-o", str(temporary)], check=True)
        temporary.replace(pdf)
    return pdf


def paragraphs(text, limit=510):
    current = []
    size = 0
    for line in text.splitlines():
        line = normalize(line)
        if not line:
            continue
        if size + len(line) > limit and current:
            yield "\n".join(current)
            current = []
            size = 0
        current.append(line)
        size += len(line)
    if current:
        yield "\n".join(current)


def numcell(text):
    return bool(re.fullmatch(r"[-+−]?[\d,]+(?:\.\d+)?%?", text))


def ruled_tables(page):
    tables = page.find_tables().tables
    # Financial statements sometimes use borderless columns.
    if not tables and re.search(r"营业(总)?收入|归属于.{0,12}净利润", page.get_text()):
        tables = page.find_tables(vertical_strategy="text", horizontal_strategy="text",
                                  min_words_vertical=3, min_words_horizontal=2).tables
    return tables


def annual_table_rows(raw):
    """Recover wrapped labels in the early annual financial summary from visual lines."""
    lines = raw.splitlines()
    columns = []
    for index, original in enumerate(lines):
        line = unicodedata.normalize("NFKC", original).strip()
        if "第一季度" in line or "非经常性损益项目和金额" in line:
            columns = []
        if re.search(r"2024\s*年", line) and re.search(r"2023\s*年", line):
            columns = [m[0] for m in ANNUAL_COLUMN.finditer(line)]
            if not columns or not columns[0].startswith("2024"):
                columns = []
            continue
        entered = list(FINANCIAL_NUMBER.finditer(line))
        if not columns or len(entered) < 3:
            continue
        before = line[:entered[0].start()].strip()
        after = line[entered[-1].end():].strip()
        label = before + after
        previous = normalize(lines[index - 1]) if index else ""
        following = normalize(lines[index + 1]) if index + 1 < len(lines) else ""
        if not before and re.match(r"归属于|经营活动", previous.replace(" ", "")):
            label = previous + label
        if re.fullmatch(r"(?:股东的)?净(?:利润|资产).{0,6}|(?:现金)?流量净额.{0,6}|[量]?净额.{0,6}|[率股]\).{0,2}",
                        following.replace(" ", "")):
            label += following
        label = label.replace(" ", "")
        match = FINANCIAL_LABEL.match(label)
        if not match or re.search(r"扣除|非经常", label):
            continue
        # Preserve nonnumeric change descriptions in their original column.
        cells = [label, entered[0][0], entered[1][0]]
        if len(columns) >= 4:
            if len(entered) >= 4:
                cells += [line[entered[1].end():entered[3].start()].strip(), *[m[0] for m in entered[3:]]]
            else:
                cells += [line[entered[1].end():entered[2].start()].strip(), entered[2][0]]
        else:
            cells += [m[0] for m in entered[2:]]
        headers = ["项目", *columns]
        if len(headers) < len(cells):
            headers += ["同年分列"] * (len(cells) - len(headers))
        cells += [""] * (len(headers) - len(cells))
        yield headers, cells


def extract(report):
    pdf = download(report)
    rows = []
    section = "封面与目录"
    with pymupdf.open(pdf) as doc:
        if len(doc) < 60:
            raise ValueError(f"{report['name']} 只有 {len(doc)} 页，请检查是否为全文")
        toc = {}
        by_page = {}
        for p in (doc[i] for i in range(min(len(doc), 12))):
            for line in p.get_text(sort=True).splitlines():
                compact = normalize(line)
                m = re.match(r"(第\s*[一二三四五六七八九十]+\s*节\s*.+?)\s*(?:[.·…]\s*){3,}(\d+)$", compact)
                if m:
                    toc[int(m[2])] = m[1].strip()
        for level, title, pno in doc.get_toc():
            title = normalize(title)
            if pno > 0 and SECTION.fullmatch(title):
                by_page[pno] = title
        by_page = {**toc, **by_page}
        for page in doc:
            pno = page.number + 1
            raw = page.get_text(sort=True)
            section = by_page.get(pno, section)
            for line in raw.splitlines():
                compact = normalize(line)
                if SECTION.fullmatch(compact):
                    section = compact
            if pno >= 5 and section == "封面与目录":
                section = "正文（章节标题未识别）"

            def add(body, kind, **fields):
                rows.append(dict(id=f"{report['code']}-P{pno:03d}-{len(rows):05d}",
                                 code=report["code"], page=pno, section=section,
                                 kind=kind, text=body, **fields))

            for part in paragraphs(raw):
                add(part, "正文")
            if 4 <= pno <= 12 and "公司简介" in section:
                for headers, cells in annual_table_rows(raw):
                    add(" | ".join(headers) + "\n" + " | ".join(cells), "表格",
                        headers=headers, cells=cells, unit="", annual=True)
            for table in ruled_tables(page):
                extracted = [[normalize(v.replace("\n", " ")) if v else "" for v in row]
                             for row in table.extract()]
                extracted = [row for row in extracted if any(row)]
                if len(extracted) < 2:
                    continue
                # Retain each cell in its original column, including blank/merged cells.
                headers = [
                    normalize(v.replace("\n", " ")) if v else "" for v in table.header.names
                ]
                if len(headers) != len(extracted[0]) or sum(bool(v) for v in headers) < 2:
                    headers = extracted[0]
                if any(numcell(v) for v in headers):
                    headers = [""] * len(extracted[0])
                near = page.get_textbox(pymupdf.Rect(0, max(0, table.bbox[1] - 85),
                                                    page.rect.width, table.bbox[1]))
                units = re.findall(r"单位\s*[:：]\s*([^\n]{1,24})", normalize(near))
                unit = units[-1].strip() if units else ""
                header = " | ".join(v or f"列{i+1}" for i, v in enumerate(headers))
                for cell_row in extracted:
                    if cell_row == headers or sum(bool(v) for v in cell_row) < 2:
                        continue
                    add(f"{header}\n{' | '.join(cell_row)}" + (f"\n单位: {unit}" if unit else ""),
                        "表格", headers=headers, cells=cell_row, unit=unit)
        summary = {"code": report["code"], "pages": len(doc), "section_markers": len(by_page),
                   "chunks": len(rows), "table_rows": sum(r["kind"] == "表格" for r in rows),
                   "sha256": hashlib.file_digest(pdf.open("rb"), "sha256").hexdigest()}
    print(report["name"], summary, flush=True)
    return rows, summary


def bucket(token):
    return int.from_bytes(hashlib.blake2s(token.encode(), digest_size=4).digest(), "little") % DIMENSIONS


def embedding(counts, idf):
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    for token, tf in counts.items():
        ix = bucket(token)
        vector[ix] += np.log1p(tf) * idf[ix]
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        extracted = list(pool.map(extract, REPORTS))
    rows = [row for batch, _ in extracted for row in batch]
    counts = [Counter(tokens(row["text"])) for row in rows]
    dfs = np.zeros(DIMENSIONS, dtype=np.float32)
    for count in counts:
        for ix in {bucket(t) for t in count}:
            dfs[ix] += 1
    idf = (1 + np.log((1 + len(rows)) / (1 + dfs))).astype(np.float32)
    matrix = np.stack([embedding(count, idf) for count in counts])
    # A fixed random projection estimates a latent semantic subspace from this corpus.
    rng = np.random.default_rng(2024)
    projected = matrix @ rng.standard_normal((DIMENSIONS, 128), dtype=np.float32)
    q, _ = np.linalg.qr(projected, mode="reduced")
    _, _, components = np.linalg.svd(q.T @ matrix, full_matrices=False)
    basis = components[:112].T.copy()
    dense = matrix @ basis
    dense /= np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-9)
    np.savez_compressed(DATA / "vectors.npz", dense=dense, basis=basis, idf=idf)
    with gzip.open(DATA / "chunks.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False)
    (DATA / "manifest.json").write_text(json.dumps([info for _, info in extracted],
                                                ensure_ascii=False, indent=2))
    print(f"已索引 {len(rows)} 个文本块")


if __name__ == "__main__":
    main()
