"""Regression checks for lossy-cleanup boundaries and auditable deduplication."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "project/medaegis/data_pipeline/02_clean_and_deduplicate.py"
SPEC = importlib.util.spec_from_file_location("basic_clean", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BasicCleanTests(unittest.TestCase):
    def test_cleaning_preserves_meaning_and_paragraphs(self):
        original = "  无发热\r\n剂量0.5 mg<BR /><br>不要加量\t保留\u0007  "
        cleaned, _ = MODULE.clean_text(original)
        self.assertEqual(cleaned, "无发热\n剂量0.5 mg\n\n不要加量\t保留")
        self.assertNotEqual(cleaned, MODULE.comparison_text(cleaned))
        self.assertEqual(MODULE.clean_text(cleaned)[0], cleaned)

    def test_candidates_are_not_removed(self):
        original = "报告https://example.com/a?id=1&not=2这是正文，电话13812345678，驻站医，�"
        self.assertEqual(MODULE.clean_text(original)[0], original)
        self.assertIn("url_candidate", MODULE.detect(original))
        self.assertIn("phone_candidate", MODULE.detect(original))

    def test_paragraphs_blocks_and_inline_text(self):
        self.assertEqual(MODULE.clean_text('<P class="answer">第一段。</P>\n<p>第二段。</p>')[0],
                         "第一段。\n\n第二段。")
        self.assertEqual(MODULE.clean_text("正文。</p>")[0], "正文。")
        self.assertEqual(MODULE.clean_text("<div><h2>标题</h2><p><strong>重点</strong>正文</p></div>")[0],
                         "标题\n\n重点正文")
        self.assertEqual(MODULE.clean_text("<p>只有一段。</p>")[0], "只有一段。")

    def test_lists(self):
        self.assertEqual(MODULE.clean_text("<ul><li>甲</li><li>乙</li></ul>")[0], "- 甲\n- 乙")
        self.assertEqual(MODULE.clean_text('<ol start="3"><li>甲</li><li value="6">乙</li></ol>')[0],
                         "3. 甲\n6. 乙")
        nested = "<ul><li>甲<ul><li>乙</li></ul></li></ul>"
        text, _, issues = MODULE.clean_field(nested)
        self.assertEqual(text, nested)
        self.assertIn("html_nested_list", issues)

    def test_links_and_query_strings_are_preserved(self):
        self.assertEqual(MODULE.clean_text('<a href="https://example.com/a?a=1&not=2">来源</a>')[0],
                         "来源 (https://example.com/a?a=1&not=2)")
        self.assertEqual(MODULE.clean_text('<p>https://example.com/a?a=1&not=2这是正文</p>')[0],
                         "https://example.com/a?a=1&not=2这是正文")
        self.assertEqual(MODULE.clean_text('<a href="https://example.com">https://example.com</a>')[0],
                         "https://example.com")

    def test_subscript_superscript_and_entities(self):
        self.assertEqual(MODULE.clean_text("维生素B<sub>1</sub>，H<sub>2</sub>O，x<sup>2</sup>")[0],
                         "维生素B₁，H₂O，x²")
        self.assertEqual(MODULE.clean_text("x<sub>ij</sub>")[0], "x_(ij)")
        self.assertEqual(MODULE.clean_text("甲&amp;乙&nbsp;丙，x &lt; 5")[0], "甲&乙\u00a0丙，x < 5")
        self.assertEqual(MODULE.clean_text("甲&#7;乙")[0], "甲乙")

    def test_entity_encoded_html_is_rendered(self):
        examples = {
            "&lt;P&gt;第一段&lt;/P&gt;&lt;p&gt;第二段&lt;/p&gt;": "第一段\n\n第二段",
            "&lt;br&gt;正文&lt;br&gt;结尾": "正文\n结尾",
            "维生素B&lt;sub&gt;1&lt;/sub&gt;": "维生素B₁",
            "<div>&lt;p&gt;甲&lt;/p&gt;&lt;p&gt;乙&lt;/p&gt;</div>": "甲\n\n乙",
            "x &lt; 5 且 y &gt; 2": "x < 5 且 y > 2",
            '&lt;a href=&quot;https://example.com/?a=1&amp;not=2&quot;&gt;来源&lt;/a&gt;':
                "来源 (https://example.com/?a=1&not=2)",
            '<a href="https://example.com/?x=&quot;value&quot;">来源</a>':
                '来源 (https://example.com/?x="value")',
            "&lt;p&gt;https://example.com/?x=1&amp;amp;value=2&lt;/p&gt;":
                "https://example.com/?x=1&amp;value=2",
        }
        for original, expected in examples.items():
            with self.subTest(original=original):
                cleaned, _, issues = MODULE.clean_field(original)
                self.assertEqual(cleaned, expected)
                self.assertEqual(issues, [])
        for original in ("&lt;table&gt;正文&lt;/table&gt;", "类型&lt;T&gt;",
                         "&amp;lt;p&amp;gt;正文&amp;lt;/p&amp;gt;"):
            cleaned, _, issues = MODULE.clean_field(original)
            self.assertEqual(cleaned, original)
            self.assertTrue(issues)

    def test_hidden_content_and_complex_html(self):
        self.assertEqual(MODULE.clean_text("甲<!-- comment --><script>bad()</script><style>.x{}</style>乙")[0], "甲乙")
        for original in ("<table><tr><td>剂量</td><td>5mg</td></tr></table>",
                         '<img src="report.png" alt="报告">', "<code>a  b</code>", "类型<T>"):
            cleaned, _, issues = MODULE.clean_field(original)
            self.assertEqual(cleaned, original)
            self.assertIn("html_complex_or_unknown_structure", issues)
        original = "开头<script>未闭合后续内容"
        cleaned, _, issues = MODULE.clean_field(original)
        self.assertEqual(cleaned, original)
        self.assertIn("html_unclosed_structure", issues)
        self.assertEqual(MODULE.clean_text("数值 < 5 且 > 2")[0], "数值 < 5 且 > 2")

    def test_general_control_policy(self):
        self.assertEqual(MODULE.clean_text("a\x00b\x07c\x1bd\x7fe\x9ff")[0], "abcdef")
        self.assertEqual(MODULE.clean_text("a\r\nb\rc\vd\fe\x85f\tg")[0], "a\nb\nc\nd\ne\nf\tg")
        # Do not remove meaningful format characters or literal escape text.
        self.assertEqual(MODULE.clean_text("a\u200db\\x07")[0], "a\u200db\\x07")
        for code in list(range(32)) + list(range(127, 160)):
            result = MODULE.clean_text("a" + chr(code) + "b")[0]
            separator = "\t" if code == 9 else "\n" if code in (10, 11, 12, 13, 133) else ""
            self.assertEqual(result, "a" + separator + "b")

    def test_ambiguous_html_is_preserved(self):
        for original in ("甲<strong>乙", "甲</sub>乙", "甲</a>乙",
                         "<b><i>正文</b></i>", "<li>无列表容器</li>",
                         '<a href="one"><a href="two">正文</a></a>'):
            cleaned, _, issues = MODULE.clean_field(original)
            self.assertEqual(cleaned, original)
            self.assertTrue(issues)

    def test_json_and_html_quarantine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "fixture.json"
            valid = dict(instruction="问题", input="", output="<table><tr><td>5mg</td></tr></table>")
            source.write_text(json.dumps(valid) + '\n' +
                              '{"instruction":"甲","input":"","output":"乙","output":"丙"}\n', encoding="utf-8")
            out = root / "out"
            subprocess.run([sys.executable, str(SCRIPT), "--input", str(source),
                            "--output-dir", str(out), "--revision", "fixture-v1"],
                           check=True, capture_output=True, text=True)
            records = [json.loads(line) for line in (out / "quarantine.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["original"], valid)
            self.assertIn("html_complex_or_unknown_structure", records[0]["reasons"])
            self.assertEqual(records[1]["reasons"], ["invalid_json"])
            self.assertEqual((out / "cleaned.jsonl").read_text(), "")

    def test_full_pipeline_partition_duplicates_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "fixture.json"
            rows = [
                dict(instruction="问题", input="", output="第一段<br /><br />第二段"),
                dict(instruction=" 问题 ", input="", output="第一段\n\n第二段"),
                dict(instruction="问题", input="", output="另一个答案"),
                dict(instruction="   ", input="", output="有回答"),
                dict(instruction="问题2", input="", output="<br>\u0007"),
                dict(instruction="问题3", input=None, output="回答"),
                dict(instruction="网址https://example.com/后面中文", input="", output="电话13812345678"),
                dict(instruction="问题损坏", input="", output="正文�缺失"),
                dict(instruction="问题�", input="", output="正常回答"),
                dict(instruction="问题实体", input="", output="正文&#xfffd;缺失"),
            ]
            source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\nnot json\n\n", encoding="utf-8")
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            out = root / "out"
            command = [sys.executable, str(SCRIPT), "--input", str(source), "--output-dir", str(out), "--revision", "fixture-v1"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            report = json.loads((out / "cleaning_report.json").read_text())
            self.assertEqual(report["counts"]["physical_lines"], 12)
            self.assertEqual(report["counts"]["retained"], 3)
            self.assertEqual(report["counts"]["duplicates"], 1)
            self.assertEqual(report["counts"]["quarantined"], 8)
            quarantined = [json.loads(line) for line in (out / "quarantine.jsonl").read_text().splitlines()]
            corrupt = [row for row in quarantined if "replacement_character" in row["reasons"]]
            self.assertEqual([row["line_no"] for row in corrupt], [8, 9, 10])
            self.assertEqual(corrupt[0]["original"]["output"], "正文�缺失")
            kept = [json.loads(line) for line in (out / "cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(kept[0]["output"], "第一段\n\n第二段")
            self.assertEqual(kept[2]["prompt_text"], rows[6]["instruction"])
            self.assertEqual(kept[2]["output"], rows[6]["output"])
            flagged = [json.loads(line) for line in (out / "flagged_records.jsonl").read_text().splitlines()]
            self.assertTrue(all("review_status" not in row for row in flagged))
            self.assertTrue(all("flags_before" not in row and "flags_after" not in row for row in flagged))
            self.assertNotIn("flags", flagged[0])  # br repaired; no remaining issues
            contact = next(row for row in flagged if row["line_no"] == 7)
            self.assertEqual(contact["flags"], ["phone_candidate", "url_candidate"])
            duplicates = [json.loads(line) for line in (out / "duplicate_records.jsonl").read_text().splitlines()]
            self.assertEqual(duplicates[0]["kept_source_id"], kept[0]["source_id"])
            groups = [json.loads(line) for line in (out / "multi_answer_groups.jsonl").read_text().splitlines()]
            self.assertEqual(len(groups), 1)
            self.assertEqual([r["line_no"] for r in groups[0]["members"]], [1, 3])
            # Refuse to overwrite a completed run.
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
