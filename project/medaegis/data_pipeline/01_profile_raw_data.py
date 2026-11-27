#!/usr/bin/env python3
"""Read-only JSONL audit. Standard library; optional local tokenizer on reservoir sample."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import time
import unicodedata


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).digest()


def normalize(text):
    # Audit-only comparison key: preserve original data, numbers, units and negation.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def quantiles(hist):
    total = sum(hist.values())
    if not total:
        return {}
    result = {}
    ordered = sorted(hist.items())
    for p in (0.5, 0.9, 0.95, 0.99, 1):
        threshold = max(1, math.ceil(p * total))
        seen = 0
        for value, count in ordered:
            seen += count
            if seen >= threshold:
                result[f"p{int(p * 100)}"] = value
                break
    result["mean"] = sum(k * v for k, v in hist.items()) / total
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0=whole file; otherwise physical-line prefix for debugging")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", help="Optional local model/tokenizer directory; no network downloads")
    args = parser.parse_args()
    if args.limit < 0 or args.sample_size < 1:
        parser.error("limit must be >= 0; sample-size must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    started = time.perf_counter()
    metadata = args.input.parent.parent / ".cache/huggingface/download" / args.input.parent.name / (args.input.name + ".metadata")
    provenance = metadata.read_text().splitlines() if metadata.exists() else []
    revision = provenance[0] if provenance else "unknown"
    source_file = f"{args.input.parent.name}/{args.input.name}"
    counts = Counter()
    lengths = {k: Counter() for k in ("instruction", "input", "prompt", "output", "prompt_plus_output")}
    rules = {
        "html_tag": re.compile(r"</?[A-Za-z][^>]*>"),
        "replacement_character": re.compile("\ufffd"),
        "control_character": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"),
        "phone_candidate": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "id_candidate": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
        "email_candidate": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
        "url_candidate": re.compile(r"https?://\S+"),
    }
    keywords = ("病情分析", "指导意见", "医生询问", "建议", "治疗", "症状", "检查", "用法", "不良反应", "禁忌")
    hits = Counter()
    keyword_hits = Counter()
    examples = {key: [] for key in rules}
    sample = []
    rng = random.Random(args.seed)
    file_hash = hashlib.sha256()
    read_bytes = 0
    db = sqlite3.connect(args.output_dir / "audit.sqlite")
    db.execute("CREATE TABLE records (record_hash BLOB PRIMARY KEY, prompt_hash BLOB, n INTEGER) WITHOUT ROWID")
    sql = "INSERT INTO records VALUES (?, ?, 1) ON CONFLICT(record_hash) DO UPDATE SET n=n+1"
    batch = []
    with args.input.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if args.limit and line_no > args.limit:
                break
            counts["physical_lines"] += 1
            file_hash.update(raw)
            read_bytes += len(raw)
            if not raw.strip():
                counts["blank_lines"] += 1
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                counts["invalid_json"] += 1
                continue
            counts["parsed_json"] += 1
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ("instruction", "input", "output")):
                counts["schema_invalid"] += 1
                continue
            counts["schema_valid"] += 1
            instruction, extra, answer = (row[k] for k in ("instruction", "input", "output"))
            prompt = "\n".join(v.strip() for v in (instruction, extra) if v.strip())
            counts["empty_instruction"] += not instruction.strip()
            counts["empty_input"] += not extra.strip()
            counts["empty_prompt"] += not prompt
            counts["empty_output"] += not answer.strip()
            values = dict(instruction=instruction, input=extra, prompt=prompt, output=answer, prompt_plus_output=prompt + answer)
            for key, value in values.items():
                lengths[key][len(value)] += 1
            source_id = f"medical:{revision}:{source_file}:line:{line_no}"
            text = prompt + "\n" + answer
            flags = []
            for key, pattern in rules.items():
                if pattern.search(text):
                    hits[key] += 1
                    flags.append(key)
                    if len(examples[key]) < 10:
                        examples[key].append(source_id)
            for word in keywords:
                keyword_hits[word] += word in text
            normalized_prompt, normalized_answer = normalize(prompt), normalize(answer)
            batch.append((digest([normalized_prompt, normalized_answer]), digest(normalized_prompt)))
            if len(batch) >= 10000:
                db.executemany(sql, batch)
                db.commit()
                batch.clear()
            item = dict(source_id=source_id, line_no=line_no, instruction=instruction, input=extra, output=answer, prompt_text=prompt, quality_flags=flags, original_source="unknown")
            n = counts["schema_valid"]
            if len(sample) < args.sample_size:
                sample.append(item)
            else:
                index = rng.randrange(n)
                if index < args.sample_size:
                    sample[index] = item
            if line_no % 100000 == 0:
                print(f"scanned {line_no:,} lines in {time.perf_counter() - started:.1f}s", flush=True)
    db.executemany(sql, batch)
    db.commit()
    unique_records = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    db.execute("CREATE INDEX by_prompt ON records(prompt_hash)")
    unique_prompts, multiple_answers = db.execute("SELECT COUNT(*), COALESCE(SUM(n > 1),0) FROM (SELECT COUNT(*) n FROM records GROUP BY prompt_hash)").fetchone()
    db.close()
    sample.sort(key=lambda item: item["line_no"])
    with (args.output_dir / "sample.jsonl").open("w", encoding="utf-8") as handle:
        for item in sample:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    tokens = {"status": "not_measured", "reason": "Pass --tokenizer with the intended local tokenizer; characters are not tokens."}
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        token_lengths = {key: Counter() for key in ("prompt", "output", "prompt_plus_output")}
        for item in sample:
            p = len(tokenizer.encode(item["prompt_text"], add_special_tokens=False))
            a = len(tokenizer.encode(item["output"], add_special_tokens=False))
            for key, value in (("prompt", p), ("output", a), ("prompt_plus_output", p + a)):
                token_lengths[key][value] += 1
        tokens = dict(status="sample_estimate", sample_size=len(sample), tokenizer=args.tokenizer,
                      includes_chat_template=False, quantiles={k: quantiles(v) for k, v in token_lengths.items()})
    seconds = time.perf_counter() - started
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), input=str(args.input.resolve()),
        repository="shibing624/medical", revision=revision, download_metadata=provenance,
        scope="full_file" if not args.limit else "debug_prefix", limit=args.limit,
        file_bytes=args.input.stat().st_size, scanned_bytes=read_bytes,
        sha256_scanned_bytes=file_hash.hexdigest(),
        sha256_matches_download_etag=(file_hash.hexdigest() == provenance[1]) if not args.limit and len(provenance) > 1 and len(provenance[1]) == 64 else None,
        counts=dict(counts), character_lengths={k: quantiles(v) for k, v in lengths.items()},
        normalized_exact_duplicates=dict(normalization="NFC + whitespace collapse; question-answer key",
            scope="schema_valid_rows_only", unique_records=unique_records,
            duplicate_excess=counts["schema_valid"] - unique_records, unique_prompts=unique_prompts,
            prompts_with_multiple_distinct_answers=multiple_answers),
        rule_hits=dict(hits), rule_hit_denominator=counts["schema_valid"], rule_examples=examples,
        keyword_document_hits=dict(keyword_hits), keyword_note="Preset keyword presence per record, not segmented word frequency.",
        sample_size=len(sample), sample_seed=args.seed, sample_method="reservoir over schema-valid rows",
        token_lengths=tokens, elapsed_seconds=seconds,
        lines_per_second=counts["physical_lines"] / max(seconds, 1e-9),
        limitations=["Flags are review candidates, not confirmed privacy or corruption.",
                     "No deletion, MinHash, semantic deduplication, cross-split checking or training performed."])
    (args.output_dir / "profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "counts": dict(counts), "elapsed_seconds": seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
