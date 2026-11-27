#!/usr/bin/env python3
"""Disk-backed MinHash/LSH candidate audit. Never removes training records.

Requires datasketch==2.0.0. Run a pilot before deciding thresholds or a full run.
All batches share one SQLite index. Input is immutable cleaned.jsonl from stage 02.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
import unicodedata


VERSION = "minhash_candidate_audit_v1"


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def shingles(text, size=3):
    text = normalize(text)
    if len(text) < size:
        return {text} if text else set()
    return {text[i:i + size] for i in range(len(text) - size + 1)}


def jaccard(left, right):
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def guards(left, right):
    """Review hints only: equality does not prove clinical equivalence."""
    patterns = {
        "numbers": r"\d+(?:\.\d+)?",
        "units": r"(?i)(?<![a-z])(?:mg|kg|ml|mmol|mmhg|cm|mm|g|l)(?![a-z])|毫克|微克|毫升|公斤|岁|天|周|月|年|℃|%",
        "negation": r"没有|未见|无|不|未|否认|阴性|阳性",
    }
    return [name + "_differ" for name, pattern in patterns.items()
            if re.findall(pattern, left) != re.findall(pattern, right)]


def band_keys(tokens, seed, bands, rows):
    from datasketch import MinHash
    sketch = MinHash(num_perm=bands * rows, seed=seed, scheme="affine32")
    ordered = sorted(tokens)
    for start in range(0, len(ordered), 512):
        sketch.update_batch([token.encode("utf-8") for token in ordered[start:start + 512]])
    return [hashlib.blake2b(i.to_bytes(4, "big") +
                sketch.hashvalues[i * rows:(i + 1) * rows].astype(">u8").tobytes(), digest_size=16).digest()
            for i in range(bands)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10000, help="Default 10K pilot; 0=full file")
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--bands", type=int, default=16)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-threshold", type=float, default=0.85)
    parser.add_argument("--answer-threshold", type=float, default=0.90)
    parser.add_argument("--max-candidates", type=int, default=1000,
                        help="Per-record recall cap; every truncation is reported")
    args = parser.parse_args()
    if min(args.shingle_size, args.bands, args.rows, args.max_candidates) < 1 or args.limit < 0:
        parser.error("Sizes must be positive and limit nonnegative")
    if not (0 <= args.prompt_threshold <= 1 and 0 <= args.answer_threshold <= 1):
        parser.error("Thresholds must be in [0, 1]")
    if version("datasketch") != "2.0.0":
        parser.error("Use datasketch==2.0.0 to freeze signature scheme and reproducibility")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / ".gitignore").write_text("*\n!.gitignore\n")
    initial = args.input.stat()
    started = time.perf_counter()
    source_hash = hashlib.sha256()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    counts = Counter({name: 0 for name in ("input_records", "lsh_candidates_checked", "prompt_similar_pairs",
        "qa_similar_pairs", "same_prompt_pairs", "truncated_queries", "empty_prompt_skipped")})
    db = sqlite3.connect(args.output_dir / "lsh.sqlite")
    db.execute("CREATE TABLE records(id INTEGER PRIMARY KEY, source_id TEXT UNIQUE NOT NULL, offset INTEGER NOT NULL)")
    db.execute("CREATE TABLE buckets(bucket BLOB NOT NULL, id INTEGER NOT NULL, PRIMARY KEY(bucket,id)) WITHOUT ROWID")
    offset = 0
    try:
        with args.input.open("rb") as source, args.input.open("rb") as lookup, \
             (args.output_dir / "candidate_pairs.jsonl").open("w", encoding="utf-8") as pairs, \
             (args.output_dir / "truncated_queries.jsonl").open("w", encoding="utf-8") as truncated:
            def emit(handle, item):
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

            for number, raw in enumerate(source, 1):
                if args.limit and number > args.limit:
                    break
                source_hash.update(raw)
                row = json.loads(raw)
                if not all(isinstance(row.get(k), str) for k in ("source_id", "prompt_text", "output")):
                    raise ValueError(f"Invalid stage-02 schema at line {number}")
                q = normalize(row["prompt_text"])
                tokens = shingles(q, args.shingle_size)
                counts["input_records"] += 1
                db.execute("INSERT INTO records VALUES(?,?,?)", (number, row["source_id"], offset))
                offset += len(raw)
                if not tokens:
                    counts["empty_prompt_skipped"] += 1
                    continue
                keys = band_keys(tokens, args.seed, args.bands, args.rows)
                placeholders = ",".join("?" for _ in keys)
                ids = [x[0] for x in db.execute(
                    f"SELECT DISTINCT id FROM buckets WHERE bucket IN ({placeholders}) ORDER BY id LIMIT ?",
                    (*keys, args.max_candidates + 1))]
                if len(ids) > args.max_candidates:
                    counts["truncated_queries"] += 1
                    emit(truncated, dict(source_id=row["source_id"], limit=args.max_candidates,
                                         reason="lsh_query_cap", recall_incomplete=True))
                answer_tokens = None
                for previous in ids[:args.max_candidates]:
                    sid, position = db.execute("SELECT source_id,offset FROM records WHERE id=?", (previous,)).fetchone()
                    lookup.seek(position)
                    other = json.loads(lookup.readline())
                    counts["lsh_candidates_checked"] += 1
                    other_q = normalize(other["prompt_text"])
                    similarity = jaccard(tokens, shingles(other_q, args.shingle_size))
                    if similarity < args.prompt_threshold:
                        continue
                    counts["prompt_similar_pairs"] += 1
                    if answer_tokens is None:
                        answer_tokens = shingles(row["output"], args.shingle_size)
                    answer_similarity = jaccard(answer_tokens, shingles(other["output"], args.shingle_size))
                    same_prompt = q == other_q
                    counts["same_prompt_pairs"] += same_prompt
                    qa_similar = answer_similarity >= args.answer_threshold
                    counts["qa_similar_pairs"] += qa_similar
                    # Include full texts for review; never infer correctness or deletion.
                    emit(pairs, dict(left_source_id=sid, right_source_id=row["source_id"],
                        left_prompt=other["prompt_text"], right_prompt=row["prompt_text"],
                        left_answer=other["output"], right_answer=row["output"],
                        prompt_jaccard=similarity, answer_jaccard=answer_similarity,
                        same_prompt=same_prompt, qa_similarity_threshold_met=qa_similar,
                        prompt_guards=guards(other_q, q), answer_guards=guards(other["output"], row["output"]),
                        action="review_only"))
                db.executemany("INSERT INTO buckets VALUES(?,?)", [(key, number) for key in keys])
                if number % 1000 == 0:
                    db.commit()
                    print(f"indexed {number:,}; pairs {counts['prompt_similar_pairs']:,}; {time.perf_counter()-started:.1f}s", flush=True)
        db.commit()
    finally:
        db.close()
    current = args.input.stat()
    if (initial.st_size, initial.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
        raise RuntimeError("Input changed during indexing")
    elapsed = time.perf_counter() - started
    report = dict(status="complete", version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        scope="full_file" if args.limit == 0 else "debug_prefix", counts=dict(counts),
        input=str(args.input.resolve()), sha256_scanned_bytes=source_hash.hexdigest(), scanned_bytes=offset,
        input_file_bytes=initial.st_size, script_sha256=script_hash,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        datasketch_version=version("datasketch"), scheme="affine32", seed=args.seed,
        signature_size=args.bands * args.rows, elapsed_seconds=elapsed,
        records_per_second=counts["input_records"] / max(elapsed, 1e-9),
        database_bytes=(args.output_dir / "lsh.sqlite").stat().st_size,
        records_removed=0,
        limitations=["LSH is approximate; exact Jaccard only verifies recalled pairs, not recall completeness.",
            "Query caps may omit additional pairs; inspect truncated_queries before full use.",
            "Prefix pilot is not a representative quality estimate or formal full-pool result.",
            "Shingle sets discard order and frequency; clinical equivalence requires further review.",
            "Numeric/unit/negation hints are incomplete; empty hints do not prove clinical equivalence.",
            "No automatic deletion, connected-component grouping or semantic deduplication.",
            "A long running job does not resume; use a new output directory after failure."])
    (args.output_dir / "minhash_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output_dir=str(args.output_dir), counts=dict(counts)), ensure_ascii=False))


if __name__ == "__main__":
    main()
