"""Run after installing datasketch==2.0.0 in the project environment."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "project/medaegis/data_pipeline/02b_minhash_candidates.py"
SPEC = importlib.util.spec_from_file_location("minhash_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MinHashTests(unittest.TestCase):
    def test_character_shingles(self):
        self.assertEqual(MODULE.shingles("持续发热", 3), {"持续发", "续发热"})
        self.assertEqual(MODULE.shingles("短", 3), {"短"})
        self.assertEqual(MODULE.shingles("", 3), set())

    def test_exact_jaccard(self):
        self.assertAlmostEqual(MODULE.jaccard({"a", "b", "c"}, {"b", "c", "d"}), 0.5)
        self.assertEqual(MODULE.jaccard({"a"}, {"b"}), 0)

    def test_guards(self):
        self.assertIn("numbers_differ", MODULE.guards("年龄20岁", "年龄80岁"))
        self.assertIn("negation_differ", MODULE.guards("无发热", "发热"))
        self.assertIn("units_differ", MODULE.guards("5mg", "5g"))

    def test_fixed_seed_signatures(self):
        tokens = MODULE.shingles("问题相同且文本完整")
        self.assertEqual(MODULE.band_keys(tokens, 42, 16, 8), MODULE.band_keys(tokens, 42, 16, 8))

    def test_global_candidates_caps_and_no_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "cleaned.jsonl"
            records = [dict(source_id=f"record_{i}", prompt_text="这是同一个完整的问题，用于检查不同答案不会被删除",
                            output=f"第{i}个不同的回答") for i in range(4)]
            source.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
            original = source.read_bytes()
            out = root / "out"
            command = [sys.executable, str(SCRIPT), "--input", str(source), "--output-dir", str(out),
                       "--limit", "0", "--max-candidates", "1"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads((out / "minhash_report.json").read_text())
            self.assertEqual(report["counts"]["input_records"], 4)
            self.assertEqual(report["counts"]["truncated_queries"], 2)
            self.assertEqual(report["records_removed"], 0)
            pairs = [json.loads(x) for x in (out / "candidate_pairs.jsonl").read_text().splitlines()]
            self.assertEqual(len(pairs), 3)
            for pair in pairs:
                self.assertEqual(pair["prompt_jaccard"], 1)
                self.assertEqual(pair["action"], "review_only")
                self.assertEqual(pair["left_source_id"], "record_0")
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse((out / "cleaned.jsonl").exists())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
