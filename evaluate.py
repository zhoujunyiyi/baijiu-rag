"""Replay all 10 assignment questions and record every recalled chunk."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from build import ROOT, normalize
from search import Search


def compact(s):
    return re.sub(r"\s+", "", normalize(s))


def score(test, result):
    refs = {r["id"]: r for group in result["retrieval"].values() for r in group}
    answer = defaultdict(list)
    for line in result["answer"]:
        match = re.search(r" \[([^\]]+)\]$", line)
        if match:
            answer[match[1]].append(line[:match.start()])
    checks = []
    for gold in test["gold"]:
        correct_page = {
            rid: ref for rid, ref in refs.items()
            if ref["code"] == gold["code"] and ref["page"] == gold["page"]
        }
        terms = gold.get("facts") or [gold["value"]]
        parts = []
        for term in terms:
            # The answer includes units added from cell labels. Evidence may store them separately.
            needle = compact(term)
            evidence_needle = needle.rstrip("%") if needle.endswith("%") else re.sub(r"元(?:/股)?$", "", needle)
            if "facts" in gold:
                evidence_needle = needle
            matching = [rid for rid, ref in correct_page.items()
                        if evidence_needle in compact(ref["text"])]
            cited = [rid for rid in matching if any(
                needle in compact(line) and ("metric" not in gold or f"｜{gold['metric']}：" in line)
                for line in answer[rid]
            )]
            parts.append({"target":term, "recalled":matching, "cited_correct":cited})
        checks.append({"source":gold, "checks":parts})
    rankings = [{"target":r, "correct":r in result["answer"]} for r in test.get("ranking", [])]
    flags = [bool(c["cited_correct"]) for row in checks for c in row["checks"]]
    flags += [r["correct"] for r in rankings]
    status = "正确" if all(flags) else "部分正确" if any(flags) else "未答对"
    return {"status":status, "targets_correct":sum(flags), "targets_total":len(flags),
            "checks":checks, "rankings":rankings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation.json")
    args = parser.parse_args()
    engine = Search()
    tests = json.loads((ROOT / "questions.json").read_text())
    runs = []
    for test in tests:
        result = engine.ask(test["question"])
        grade = score(test, result)
        runs.append({**test, **result, **grade})
        print(test["id"], grade["status"], f"{grade['targets_correct']}/{grade['targets_total']}",
              "|", " / ".join(result["answer"]))
    args.out.write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
