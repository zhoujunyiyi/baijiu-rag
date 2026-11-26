"""Hybrid retrieval and evidence-based extractive answers."""
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
import json
import math
import re

import numpy as np

from build import DATA, REPORTS, embedding, normalize, tokens

METRICS = {
    "营业收入": r"营业(总)?收入",
    "归母净利润": r"归(?:属于|属).{0,20}(?:股东|所有者)的净利润|归母净利润",
    "经营现金流量净额": r"经营活动产生的现金流量净额|经营现金流量净额",
    "基本每股收益": r"基本每股收益",
    "加权平均净资产收益率": r"加权平均净资产收益率",
    "归母净资产": r"归(?:属于|属).{0,20}(?:股东|所有者)的净资产|归母净资产",
    "总资产": r"总资产|资产总计",
}
NUMBER = re.compile(r"[-+−]?[\d,]+(?:\.\d+)?%?")


class Search:
    def __init__(self):
        self.reports = {r["code"]: r for r in REPORTS}
        with gzip.open(DATA / "chunks.json.gz", "rt") as stream:
            self.rows = json.load(stream)
        with np.load(DATA / "vectors.npz") as arrays:
            self.dense, self.basis, self.idf = (arrays[k] for k in ("dense", "basis", "idf"))
        self.by_code = defaultdict(list)
        self.postings = defaultdict(list)
        lengths = []
        for ix, row in enumerate(self.rows):
            self.by_code[row["code"]].append(ix)
            counts = Counter(tokens(row["text"]))
            lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self.postings[token].append((ix, tf))
        self.lengths = np.array(lengths, dtype=np.float32)
        self.avg = self.lengths.mean()

    def retrieve(self, query, code=None, metric=None, limit=5):
        query_terms = Counter(tokens(query))
        vector = embedding(query_terms, self.idf) @ self.basis
        vector /= max(np.linalg.norm(vector), 1e-9)
        cosine = self.dense @ vector
        bm25 = np.zeros(len(self.rows), dtype=np.float32)
        for term in query_terms:
            posting = self.postings.get(term, ())
            if not posting:
                continue
            ix, tf = map(np.array, zip(*posting))
            idf = math.log1p((len(self.rows) - len(ix) + .5)/(len(ix) + .5))
            bm25[ix] += idf * (tf * 2.2) / (tf + 1.2 * (.25 + .75 * self.lengths[ix]/self.avg))

        candidates = np.asarray(self.by_code[code] if code else range(len(self.rows)))
        # Reciprocal rank fusion with explicit bonuses for verified financial row labels.
        orders = [candidates[np.argsort(-score[candidates], kind="stable")] for score in (bm25, cosine)]
        fused = defaultdict(float)
        for order in orders:
            for rank, ix in enumerate(order[:100], 1):
                fused[ix] += 1/(35 + rank)
        if metric:
            for ix in fused:
                row = self.rows[ix]
                if financial_value(row, metric):
                    fused[ix] += .035 if row["page"] <= 15 else .013
                    if is_annual(row):
                        fused[ix] += .04
        best = sorted(fused, key=lambda ix: (-fused[ix], ix))[:limit]
        return [self.reference(ix, bm25[ix], cosine[ix], fused[ix]) for ix in best]

    def reference(self, ix, bm25, cosine, fused):
        row = self.rows[ix]
        report = self.reports[row["code"]]
        return {**row, "company": report["name"], "url": report["url"] + f"#page={row['page']}",
                "bm25": round(float(bm25), 3), "cosine": round(float(cosine), 4),
                "fused": round(float(fused), 5)}

    def ask(self, question):
        query = normalize(question).replace(" ", "")
        codes = []
        for report in REPORTS:
            positions = [query.find(name) for name in [report["name"], report["code"], *report.get("aliases", [])]]
            positions = [p for p in positions if p >= 0]
            if positions:
                codes.append((min(positions), report["code"]))
        companies = [code for _, code in sorted(codes)]
        mentioned = []
        for name, pattern in METRICS.items():
            match = re.search(f"(?:{pattern}|{re.escape(name)})", query)
            if match:
                mentioned.append((match.start(), name))
        metric_names = [name for _, name in sorted(mentioned)]
        hits, evidence, answers = {}, [], []
        suitable = not re.search(r"20(?:2[0-3]|2[5-9])年|同比|增速|增长率|降幅|增长多少|较上年", query)
        if companies and metric_names and suitable:
            values = defaultdict(list)
            for code in companies:
                for metric in metric_names:
                    refs = self.retrieve(f"主要会计数据和财务指标 2024年 本报告期 {metric}",
                                         code, metric, limit=5)
                    hits[f"{self.reports[code]['name']} · {metric}"] = refs
                    ref = next((r for r in refs if financial_value(r, metric) and is_annual(r)), None)
                    if ref is None:
                        answers.append(f"{self.reports[code]['name']}｜{metric}：召回结果缺少可确认的本期表格。")
                        continue
                    value, unit = financial_value(ref, metric)
                    evidence.append(ref["id"])
                    values[metric].append((Decimal(value.replace(",", "").replace("−", "-").rstrip("%")),
                                           ref["company"], unit))
                    answers.append(f"{ref['company']}｜{metric}：{value}{unit} [{ref['id']}]")
            if len(companies) > 1:
                for metric, entries in values.items():
                    if len(entries) == len(companies) and len({e[2] for e in entries}) == 1:
                        answers.append(f"{metric}从高到低：" + "、".join(e[1] for e in sorted(entries, reverse=True)) + "。")
        else:
            clean = question
            for report in REPORTS:
                for label in [report["name"], report["code"], *report.get("aliases", [])]:
                    clean = clean.replace(label, "")
            for code in companies or [None]:
                refs = self.retrieve(clean, code, limit=6)
                hits[self.reports[code]["name"] if code else "全部公司"] = refs
                for ref, excerpt in select_excerpts(refs, clean):
                    evidence.append(ref["id"])
                    answers.append(f"{ref['company']}｜{excerpt} [{ref['id']}]")
        return {"question": question, "answer": answers or ["本次召回缺少可引用的证据。"],
                "evidence": list(dict.fromkeys(evidence)), "retrieval": hits}


def is_annual(row):
    return row.get("annual") or (
        row["page"] <= 15 and "公司简介" in row["section"]
        and any(re.search(r"2024\s*年", h) for h in row["headers"])
        and not any("季度" in h for h in row["headers"])
    )


def financial_value(row, metric):
    if row["kind"] != "表格":
        return None
    cells = [v for v in row["cells"] if v]
    if len(cells) < 2 or not re.search(METRICS[metric], cells[0].replace(" ", "")):
        return None
    label = cells[0].replace(" ", "")
    if re.search(r"扣除|扣非|增长率|变动|同比|调整前|分地区|分产品", label):
        return None
    if re.search(r"万元|亿元", label):
        return None
    if "元" not in label and not row.get("annual") and re.search(r"万元|亿元", row["unit"]):
        return None
    value = next((v for v in cells[1:] if NUMBER.fullmatch(v)), None)
    if value is None:
        return None
    unit = "%" if metric.endswith("收益率") else "元/股" if metric == "基本每股收益" else "元"
    if unit == "%" and not value.endswith("%") and "%" not in label:
        return None
    return value.rstrip("%"), unit


def select_excerpts(refs, query):
    keys = {t for t in tokens(query) if len(t) > 1 and t not in
            {"哪些", "什么", "2024", "怎么", "年报", "公司", "报告", "是否", "主要"}}
    choices = []
    for rank, ref in enumerate(refs):
        if ref["kind"] == "表格":
            candidates = [ref["text"].replace("\n", " ／ ")]
        else:
            lines = []
            for line in ref["text"].splitlines():
                if re.match(r"^.*2024\s*年[年度]+报告(?:全文)?$|^[√□]|^\d+\s*/\s*\d+$", line):
                    continue
                if (re.match(r"^单位:|^\(?[一二三四五六七八九十0-9]+\)?[、.]\s*[^。；！？]{0,18}$", line)
                    or len(NUMBER.findall(line)) >= 3 and not re.search(r"[。；！？]", line)
                    or re.match(r"^(?:原料类别|占当期|本期销售量|渠道类型)", line)):
                    lines.append("␞")
                else:
                    lines.append(line)
            candidates = []
            for segment in "".join(lines).split("␞"):
                sentences = re.split(r"(?<=[。；！？;])", segment)
                candidates.extend(sentences)
                candidates.extend(a + b for a, b in zip(sentences, sentences[1:]) if 0 < len(b) < 160)
        for line in candidates:
            line = line.strip()
            if not 15 <= len(line) <= 350:
                continue
            score = sum(key in line for key in keys)
            if score >= min(2, len(keys)) and score:
                choices.append((score - len(line) / 450, -rank, ref, line))
    chosen = []
    for _, _, ref, line in sorted(choices, key=lambda r: (-r[0], -r[1])):
        if any(line in existing or existing in line or overlap(line, existing) > .72
               for _, existing in chosen):
            continue
        chosen.append((ref, line))
        if len(chosen) == 2:
            break
    return chosen


def overlap(first, second):
    a, b = (set(t for t in tokens(line) if len(t) > 1) for line in (first, second))
    return len(a & b) / max(1, len(a | b))
