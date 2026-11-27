#!/usr/bin/env python3
"""Stage 1: conservative JSONL cleanup and global exact deduplication, not MinHash.

Original files are read-only. Output is an intermediate pool, not train-ready data.
Phone numbers, URLs and different answers are preserved for subsequent review.
Standard-library only; existing output directories are never overwritten.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
import unicodedata


VERSION = "basic_clean_exact_v6"
FIELDS = ("instruction", "input", "output")
ENTITY = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);")
BARE_AMP = re.compile(r"&(?!(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);)")
# Unicode Cc consists of C0, DEL and C1. Preserve TAB/LF; normalize
# CR, vertical tab, form feed and NEL to LF before removing other controls.
CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
FLAGS = {
    "html_tag": re.compile(r"</?[A-Za-z][^>]*>"),
    "replacement_character": re.compile("\ufffd"),
    "control_character": CONTROLS,
    "phone_candidate": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_candidate": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "email_candidate": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    # Detection only! This span can include adjacent Chinese text.
    "url_candidate": re.compile(r"https?://\S+"),
}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).digest()


def comparison_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


class QAHTMLParser(HTMLParser):
    """Conservative plain-text rendering, not a browser or a universal HTML sanitizer.

    Complex or unknown structures cause the entire field to be preserved and flagged;
    the caller quarantines the whole QA.
    Sub/sup notation retains its semantic distinction (H₂O, x², or _(text)/^(text)).
    """
    BLOCKS = {"p", "div", "section", "article", "header", "footer", "main", "aside",
              "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "address", "figure", "figcaption"}
    INLINE = {"span", "b", "strong", "i", "em", "u", "small", "mark", "abbr", "cite", "font"}
    WRAPPERS = {"html", "body"}
    HIDDEN = {"script", "style", "head"}
    SUPPORTED = BLOCKS | INLINE | WRAPPERS | HIDDEN | {"br", "hr", "ul", "ol", "li", "a", "sub", "sup"}
    SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
    SUP = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.pending_breaks = 0
        self.hidden = []
        self.lists = []
        self.anchors = []
        self.annotation = None
        self.inline_stack = []
        self.issues = set()

    def add_text(self, text):
        if not text:
            return
        if self.pending_breaks:
            text = text.lstrip()
            if not text:
                return
            # Paragraph boundaries replace trailing source indentation/newlines.
            while self.parts and not self.parts[-1].strip():
                self.parts.pop()
            if self.parts:
                self.parts[-1] = self.parts[-1].rstrip()
                self.parts.append("\n" * self.pending_breaks)
            self.pending_breaks = 0
        self.parts.append(text)

    def boundary(self, count=2):
        self.pending_breaks = max(self.pending_breaks, count)

    def handle_starttag(self, tag, attrs):
        if self.hidden:
            if tag in self.HIDDEN:
                self.hidden.append(tag)
            return
        if tag in self.HIDDEN:
            self.hidden.append(tag)
            return
        if tag not in self.SUPPORTED:
            self.issues.add("html_complex_or_unknown_structure")
            return
        if self.annotation:
            self.issues.add("html_nested_annotation")
            return
        attrs = dict(attrs)
        if tag in self.INLINE | self.WRAPPERS:
            self.inline_stack.append(tag)
        if tag in self.BLOCKS or tag == "hr":
            self.boundary()
        elif tag == "br":
            self.pending_breaks = min(2, self.pending_breaks + 1)
        elif tag in {"ul", "ol"}:
            self.boundary()
            if self.lists:
                # Flat text markers alone do not faithfully retain nesting.
                self.issues.add("html_nested_list")
            try:
                start = int(attrs.get("start", "1")) if tag == "ol" else 1
            except (TypeError, ValueError):
                self.issues.add("html_invalid_list_number")
                start = 1
            if "reversed" in attrs:
                self.issues.add("html_reversed_list")
            self.lists.append([tag, start])
        elif tag == "li":
            if not self.lists:
                self.issues.add("html_orphan_list_item")
            self.boundary(1)
            marker = "- "
            if self.lists and self.lists[-1][0] == "ol":
                try:
                    number = int(attrs.get("value", self.lists[-1][1]))
                except (TypeError, ValueError):
                    self.issues.add("html_invalid_list_number")
                    number = self.lists[-1][1]
                marker = f"{number}. "
                self.lists[-1][1] = number + 1
            self.add_text(marker)
        elif tag == "a":
            if self.anchors:
                self.issues.add("html_nested_link")
            self.anchors.append((attrs.get("href") or "", len(self.parts)))
        elif tag in {"sub", "sup"}:
            self.annotation = (tag, [])

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if self.annotation:
            kind, pieces = self.annotation
            if tag != kind:
                self.issues.add("html_nested_annotation")
                return
            content = "".join(pieces)
            self.annotation = None
            if content and all(c in "0123456789+-=()" for c in content):
                self.add_text(content.translate(self.SUB if kind == "sub" else self.SUP))
            else:
                self.add_text(("_" if kind == "sub" else "^") + "(" + content + ")")
            return
        if tag not in self.SUPPORTED:
            self.issues.add("html_complex_or_unknown_structure")
        elif tag in self.INLINE | self.WRAPPERS:
            if self.inline_stack and self.inline_stack[-1] == tag:
                self.inline_stack.pop()
            else:
                self.issues.add("html_mismatched_inline_tag")
        elif tag in {"sub", "sup"}:
            self.issues.add("html_orphan_annotation")
        elif tag in self.BLOCKS:
            self.boundary()
        elif tag == "li":
            self.boundary(1)
        elif tag in {"ul", "ol"}:
            if self.lists and self.lists[-1][0] == tag:
                self.lists.pop()
            else:
                self.issues.add("html_unbalanced_list")
            self.boundary()
        elif tag == "a" and self.anchors:
            href, start = self.anchors.pop()
            label = "".join(self.parts[start:]).strip()
            if href and href != label:
                self.add_text(" (" + href + ")")
        elif tag == "a":
            self.issues.add("html_orphan_link")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in {"br", "hr"}:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if self.hidden:
            return
        if self.annotation:
            self.annotation[1].append(data)
        else:
            self.add_text(data)

    def handle_comment(self, data):
        pass

    def finish(self):
        self.close()
        if self.hidden or self.annotation or self.anchors or self.lists or self.inline_stack:
            self.issues.add("html_unclosed_structure")
        return "".join(self.parts).strip()


def has_html_markup(text):
    return bool(FLAGS["html_tag"].search(text) or "<!--" in text or re.search(r"<!doctype\b", text, re.I))


def render_html(text, decode_entities=True):
    renderer = QAHTMLParser()
    try:
        # HTMLParser accepts some semicolon-less entities. Protect literal '&not='
        # in URL query strings so it cannot silently become a Unicode character.
        # After the initial entity decode, protect EVERY ampersand: a second render
        # must not repeatedly decode URLs or escaped literal entity text.
        renderer.feed(BARE_AMP.sub("&amp;", text) if decode_entities else text.replace("&", "&amp;"))
        converted = renderer.finish()
    except (ValueError, AssertionError):
        return text, ["html_parse_error"]
    if renderer.issues:
        return text, sorted(renderer.issues)
    return converted, []


def convert_html(text):
    """Decode one entity layer, then render any markup revealed by that decode.

    A second rendering never decodes entities again. Unknown/deeper markup is
    preserved in the original field and quarantined by the caller.
    """
    if has_html_markup(text):
        converted, issues = render_html(text)
        if issues:
            return text, issues
    else:
        converted = ENTITY.sub(lambda match: unescape(match.group()), text)
    if has_html_markup(converted):
        converted, issues = render_html(converted, decode_entities=False)
        if issues:
            return text, issues
    if has_html_markup(converted):
        return text, ["html_remaining_markup"]
    # Do not recursively unescape multiple encodings; flag for separate review.
    if re.search(r"&lt;\s*/?[A-Za-z][^\n]*?&gt;", converted, re.I):
        return text, ["html_multiple_encoding_layers"]
    return converted, []


def clean_field(text):
    """Return clean text, modification names and unresolved conversion flags."""
    actions = []
    steps = (
        ("nfc", lambda s: unicodedata.normalize("NFC", s)),
        ("newline_to_lf", lambda s: s.replace("\r\n", "\n").translate(
            {ord(c): "\n" for c in "\r\v\f\x85"})),
        ("remove_non_layout_controls", lambda s: CONTROLS.sub("", s)),
        ("trim_outer_whitespace", str.strip),
    )
    for name, transform in steps:
        updated = transform(text)
        if updated != text:
            actions.append(name)
        text = updated
    rendered, issues = convert_html(text)
    if rendered != text:
        actions.append("html_to_text")
    # Entity decoding can reveal control characters; apply the same policy again.
    final = rendered.replace("\r\n", "\n").translate({ord(c): "\n" for c in "\r\v\f\x85"})
    if final != rendered and "newline_to_lf" not in actions:
        actions.append("newline_to_lf")
    without_controls = CONTROLS.sub("", final)
    if without_controls != final and "remove_non_layout_controls" not in actions:
        actions.append("remove_non_layout_controls")
    final = without_controls.strip()
    if final != without_controls and "trim_outer_whitespace" not in actions:
        actions.append("trim_outer_whitespace")
    return final, actions, issues


def clean_text(text):
    cleaned, actions, _ = clean_field(text)
    return cleaned, actions


def prompt(row):
    return "\n".join(row[k].strip() for k in ("instruction", "input") if row[k].strip())


def detect(text):
    return [name for name, pattern in FLAGS.items() if pattern.search(text)]


def reject_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", help="Defaults to local Hugging Face download metadata")
    parser.add_argument("--limit", type=int, default=0, help="0=full file; prefix for debugging only")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    metadata = args.input.parent.parent / ".cache/huggingface/download" / args.input.parent.name / (args.input.name + ".metadata")
    provenance = metadata.read_text().splitlines() if metadata.exists() else []
    revision = args.revision or (provenance[0] if provenance else None)
    if not revision:
        parser.error("No provenance metadata found; pass --revision explicitly")
    initial_stat = args.input.stat()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    started = time.perf_counter()
    source_file = f"{args.input.parent.name}/{args.input.name}"
    prefix = f"medical:{revision}:{source_file}:line:"
    counters = Counter({k: 0 for k in (
        "physical_lines", "schema_valid", "retained", "duplicates", "quarantined",
        "flagged_records", "modified_records", "multi_answer_groups", "multi_answer_records")})
    original_hits, remaining_hits, modifications, quarantine_reasons = (Counter() for _ in range(4))
    raw_hash = hashlib.sha256()
    read_bytes = 0
    db = sqlite3.connect(args.output_dir / "dedup.sqlite")
    # One global disk-backed index across all input batches. Text never enters an all-pairs matrix.
    db.execute("CREATE TABLE records(record_hash BLOB PRIMARY KEY, prompt_hash BLOB NOT NULL, line_no INTEGER NOT NULL, byte_offset INTEGER NOT NULL) WITHOUT ROWID")
    outputs = ["cleaned", "flagged_records", "quarantine", "duplicate_records", "modifications", "multi_answer_groups"]
    try:
        with ExitStack() as stack:
            handles = {name: stack.enter_context((args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8")) for name in outputs}
            def emit(name, row):
                handles[name].write(dumps(row) + "\n")

            handle = stack.enter_context(args.input.open("rb"))
            for line_no, raw in enumerate(handle, 1):
                if args.limit and line_no > args.limit:
                    break
                byte_offset = read_bytes
                read_bytes += len(raw)
                raw_hash.update(raw)
                counters["physical_lines"] += 1
                sid = prefix + str(line_no)
                error = None
                try:
                    row = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
                except (ValueError, UnicodeDecodeError):
                    error = "blank_line" if not raw.strip() else "invalid_json"
                if error is None:
                    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in FIELDS):
                        error = "schema_invalid"
                    elif any(re.search(r"[\ud800-\udfff]", row[k]) for k in FIELDS):
                        error = "unpaired_surrogate"
                if error:
                    counters["quarantined"] += 1
                    quarantine_reasons[error] += 1
                    # Byte offset plus immutable source hash permits exact recovery even of invalid UTF-8.
                    emit("quarantine", dict(source_id=sid, line_no=line_no, byte_offset=byte_offset,
                                             byte_length=len(raw), reasons=[error]))
                    continue
                counters["schema_valid"] += 1
                original = {k: row[k] for k in FIELDS}
                before_flags = detect(prompt(original) + "\n" + original["output"])
                original_hits.update(before_flags)
                cleaned, changes, html_issues = {}, {}, set()
                for field in FIELDS:
                    cleaned[field], actions, issues = clean_field(original[field])
                    html_issues.update(issues)
                    if actions:
                        changes[field] = actions
                q = prompt(cleaned)
                after_flags = sorted(set(detect(q + "\n" + cleaned["output"])) | html_issues)
                remaining_hits.update(after_flags)
                if changes:
                    counters["modified_records"] += 1
                    modifications.update(set(action for actions in changes.values() for action in actions))
                    emit("modifications", dict(source_id=sid, line_no=line_no, changes=changes))
                reasons = []
                # Preserve complex/ambiguous HTML for later processing, rather than
                # treating raw markup as plain text that already passed cleanup.
                reasons.extend(sorted(html_issues))
                # Conservative policy for this medical QA corpus: missing original
                # characters cannot be reliably reconstructed. Exclude the whole QA,
                # not just the replacement glyph. Keep the original in quarantine.
                if "replacement_character" in before_flags or "replacement_character" in after_flags:
                    reasons.append("replacement_character")
                if not q:
                    reasons.append("empty_prompt_after_cleaning")
                if not cleaned["output"]:
                    reasons.append("empty_output_after_cleaning")
                if before_flags or after_flags:
                    counters["flagged_records"] += 1
                    candidate = dict(source_id=sid, line_no=line_no, original=original, cleaned=cleaned)
                    if after_flags:
                        candidate["flags"] = after_flags
                    emit("flagged_records", candidate)
                if reasons:
                    counters["quarantined"] += 1
                    quarantine_reasons.update(reasons)
                    emit("quarantine", dict(source_id=sid, line_no=line_no,
                                             original=original, cleaned=cleaned, reasons=reasons))
                    continue
                ph = digest(comparison_text(q))
                rh = digest([comparison_text(q), comparison_text(cleaned["output"])])
                inserted = db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?)", (rh, ph, line_no, byte_offset)).rowcount
                if inserted:
                    counters["retained"] += 1
                    item = dict(source_id=sid, line_no=line_no, **cleaned, prompt_text=q,
                                prompt_hash=ph.hex(), record_hash=rh.hex())
                    if after_flags:
                        item["flags"] = after_flags
                    emit("cleaned", item)
                else:
                    counters["duplicates"] += 1
                    kept_line = db.execute("SELECT line_no FROM records WHERE record_hash=?", (rh,)).fetchone()[0]
                    emit("duplicate_records", dict(source_id=sid, kept_source_id=prefix + str(kept_line),
                        record_hash=rh.hex(), reason="same_normalized_clean_question_and_answer"))
                if line_no % 10000 == 0:
                    db.commit()
                if line_no % 100000 == 0:
                    print(f"processed {line_no:,} lines; retained {counters['retained']:,}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
            db.commit()
            db.execute("CREATE INDEX by_prompt ON records(prompt_hash)")
            db.commit()
            for (ph,) in db.execute("SELECT prompt_hash FROM records GROUP BY prompt_hash HAVING COUNT(*) > 1 ORDER BY prompt_hash"):
                members = [dict(source_id=prefix + str(n), line_no=n, byte_offset=offset, record_hash=rh.hex())
                           for n, offset, rh in db.execute("SELECT line_no,byte_offset,record_hash FROM records WHERE prompt_hash=? ORDER BY line_no", (ph,))]
                counters["multi_answer_groups"] += 1
                counters["multi_answer_records"] += len(members)
                emit("multi_answer_groups", dict(prompt_hash=ph.hex(), members=members,
                    interpretation="distinct_answer_texts_not_necessarily_conflicts", split_policy="keep_same_prompt_group_together"))
    finally:
        db.close()
    if counters["physical_lines"] != counters["retained"] + counters["duplicates"] + counters["quarantined"]:
        raise RuntimeError("Input/output partition mismatch; do not use these outputs")
    final_stat = args.input.stat()
    unchanged = (initial_stat.st_size, initial_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns)
    expected = provenance[1] if len(provenance) > 1 and re.fullmatch(r"[0-9a-f]{64}", provenance[1]) else None
    matches = raw_hash.hexdigest() == expected if expected and not args.limit else None
    verified = unchanged and matches is not False
    report = dict(status="complete" if verified else "failed_source_verification", rule_version=VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(), input=str(args.input.resolve()),
        repository="shibing624/medical", revision=revision, download_metadata=provenance,
        scope="full_file" if not args.limit else "debug_prefix", limit=args.limit,
        source_sha256=raw_hash.hexdigest(), scanned_bytes=read_bytes, file_bytes=initial_stat.st_size,
        source_unchanged_during_run=unchanged, sha256_matches_download_etag=matches,
        script_sha256=script_hash, command=sys.argv,
        counts=dict(counters), rule_hits_before=dict(original_hits), rule_hits_after=dict(remaining_hits),
        modification_record_counts=dict(modifications), quarantine_reason_counts=dict(quarantine_reasons),
        duplicate_policy="first usable record in source order; NFC + whitespace collapsed clean QA; SHA-256",
        contacts_policy="preserve_and_flag_only", url_policy="preserve_and_flag_only",
        corruption_policy="quarantine whole QA if U+FFFD appears before or after cleaning; does not detect all forms of mojibake",
        control_policy="plain-text QA: normalize CRLF/CR/VT/FF/NEL to LF; remove other C0/DEL/C1; preserve internal LF/TAB and Unicode format characters",
        html_policy="decode one entity layer then render revealed HTML without another entity decode; preserve literal URL ampersands; quarantine complex/unknown/ambiguous or remaining/multiply encoded markup; render supported paragraphs/lists/links/sub/sup",
        candidate_flags_policy="only post-clean flags, omitted when empty; raw texts and aggregate before/after counts remain available",
        stage="intermediate_not_train_ready", elapsed_seconds=time.perf_counter()-started,
        limitations=["No MinHash, semantic deduplication, cross-split leakage checks or embedding selection yet.",
            "No medical correctness, relevance, advertisement or comprehensive PII classifier.",
            "Not an HTML5 browser: complex/malformed structures may need separate processing; paragraph fragments are tolerated.",
            "Entity decoding assumes raw web text; repeatedly cleaning already-cleaned data can change escaped literals. Always run from the immutable raw input.",
            "Removing controls does not interpret terminal escape sequences or reconstruct damaged content.",
            "Flagged rows may remain in intermediate output; unresolved review is not training approval.",
            "Multi-answer groups contain IDs and byte offsets; different answers are not removed.",
            "Same-question groups must be isolated across splits in the later splitting stage.",
            "Counts of modifications and flags overlap; only retained/duplicates/quarantined partition input."])
    (args.output_dir / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "status": report["status"], "counts": dict(counters)}, ensure_ascii=False))
    if not verified:
        raise RuntimeError("Source verification failed; do not use these outputs")


if __name__ == "__main__":
    main()
