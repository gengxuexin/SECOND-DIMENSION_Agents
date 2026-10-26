"""契约与回归测试（标准库 unittest，无需 pytest）。

运行：python -m unittest discover tests -v
依赖两个模型服务已启动（python run.py serve）。
"""
from __future__ import annotations

import json
import unittest

import config
import store
from schemas import Advice, NormalizedTag, ToolError
from tools.registry import sanitize_args


class TestSchemas(unittest.TestCase):
    """结构化契约：这是 V3 可信治理的地基，必须最严。"""

    def test_advice_requires_evidence(self):
        with self.assertRaises(Exception):
            Advice(title="t", detail="d", action_type="选品",
                   confidence=0.9, tool_refs=[], evidence_ids=[])

    def test_advice_ok_with_tool_ref(self):
        a = Advice(title="t", detail="d", action_type="选品", confidence=0.9,
                   tool_refs=["query_products#sku=1"], evidence_ids=[])
        self.assertTrue(a.tool_refs)

    def test_normalized_tag_conf_range(self):
        with self.assertRaises(Exception):
            NormalizedTag(raw="a", normalized="b", conf=1.5, source="fuzzy")

    def test_tool_error_shape(self):
        e = ToolError(code="X", message="m")
        self.assertEqual(e.model_dump()["code"], "X")


class TestToolArgSanitize(unittest.TestCase):
    """工具参数白名单：防提示注入塞入越权字段。"""

    def test_blocks_injected_keys(self):
        clean = sanitize_args("query_comments",
                              {"scope": "all", "system": "evil", "prompt": "x"})
        self.assertEqual(set(clean), {"scope"})

    def test_keeps_allowed(self):
        clean = sanitize_args("analyze_comments", {"comments": ["a"], "top_n": 2})
        self.assertEqual(clean["top_n"], 2)

    def test_unknown_tool_empty(self):
        self.assertEqual(sanitize_args("evil_tool", {"a": 1}), {})


class TestTagNormalize(unittest.TestCase):
    """标签归一：别名 / 模糊 / 回落 三条路径。"""

    def test_alias(self):
        self.assertEqual(store.normalize_tag("裙子").normalized, "连衣裙")

    def test_alias_inverse(self):
        self.assertEqual(store.normalize_tag("短袖").normalized, "T恤")

    def test_exact_canonical(self):
        self.assertEqual(store.normalize_tag("连衣裙").normalized, "连衣裙")
        self.assertEqual(store.normalize_tag("连衣裙").conf, 1.0)

    def test_unknown_marked_need_confirm(self):
        nt = store.normalize_tag("完全没见过的标签xyz")
        self.assertTrue(nt.need_confirm)

    def test_empty_input(self):
        self.assertTrue(store.normalize_tag("").need_confirm)


class TestCommentSampling(unittest.TestCase):
    """均衡采样：修复"后上架品类 0 条反馈"的召回偏差。"""

    def test_balanced_covers_categories(self):
        rows = store.query_comments("all", None, 12, balanced=True)
        cats = {r["category"] for r in rows}
        self.assertGreater(len(cats), 1)

    def test_unbalanced_is_head_slice(self):
        head = store.query_comments("all", None, 4, balanced=False)
        self.assertEqual(len(head), 4)

    def test_scope_filter(self):
        rows = store.query_comments("连衣裙", None, 50)
        self.assertTrue(all(r["category"] == "连衣裙" for r in rows))


class TestServicesLive(unittest.TestCase):
    """工具层服务联通性 + 契约一致性。"""

    def test_image_service(self):
        from tools.registry import invoke
        import run as runner
        paths = runner._demo_images()
        if not paths:
            self.skipTest("无样例图片")
        out, rec = invoke("classify_image", {"image_path": paths[0], "top_k": 3})
        self.assertTrue(rec.ok, msg=out)
        self.assertLessEqual(len(out["predictions"]), 3)
        self.assertIn("model_version", out)

    def test_image_service_missing_file(self):
        from tools.registry import invoke
        out, rec = invoke("classify_image", {"image_path": "/no/such/file.jpg"})
        self.assertFalse(rec.ok)
        self.assertEqual(out["error"]["code"], "IMAGE_NOT_FOUND")

    def test_comment_service(self):
        from tools.registry import invoke
        out, rec = invoke("analyze_comments", {"comments": ["尺码偏小，建议大一码"], "top_n": 3})
        self.assertTrue(rec.ok)
        s = out["results"][0]["scores"]
        self.assertIn(s["sentiment"], ("positive", "negative", "neutral"))
        self.assertIn("尺码", s["topics"])

    def test_empty_comments_rejected(self):
        from tools.registry import invoke
        out, rec = invoke("analyze_comments", {"comments": []})
        self.assertFalse(rec.ok)


class TestGovernance(unittest.TestCase):
    """治理逻辑：证据闭环与人工分流。"""

    def test_evidence_validation_flags_orphan(self):
        from governance.trust import validate_evidence
        from schemas import CommentTheme, WorkflowState
        st = WorkflowState(task_id="t", user_query="q")
        st.themes = [CommentTheme(theme="连衣裙-尺码", category="连衣裙", count=2,
                                   evidence_ids=["NOT-EXIST"], sentiment="negative")]
        rep = validate_evidence(st)
        self.assertIn("连衣裙-尺码", rep["orphan_themes"])
        self.assertFalse(rep["hallucination_free"])

    def test_evidence_validation_passes(self):
        from governance.trust import validate_evidence
        from schemas import CommentScores, WorkflowState
        st = WorkflowState(task_id="t", user_query="q")
        st.comment_results = {"C-1": CommentScores(sentiment="negative", sentiment_conf=0.9)}
        rep = validate_evidence(st)
        self.assertTrue(rep["hallucination_free"])

    def test_review_queue_roundtrip(self):
        from governance.trust import ReviewQueue
        from schemas import NormalizedTag
        q = ReviewQueue()
        it = q.add("tag_normalize", NormalizedTag(raw="裙", normalized="裙", conf=0.3).model_dump(),
                   "低置信", "T-1")
        got = q.resolve(it["item_id"], "accept", "连衣裙")
        self.assertEqual(got["decision"], "accept")
        self.assertEqual(got["status"], "resolved")
        self.assertTrue(store.load_human_samples())


class TestWorkflowEndToEnd(unittest.TestCase):
    """端到端：状态机必须产出带证据的建议。"""

    def test_advice_flow(self):
        from orchestration import MultiAgentWorkflow
        st = MultiAgentWorkflow().run("根据评论和库存给我一些选品建议")
        self.assertIn(st.status.value, ("done", "needs_human"))
        self.assertTrue(st.comment_results, "评论洞察应产出打分结果")
        self.assertTrue(st.advice, "运营建议节点应产出建议")
        for a in st.advice:
            self.assertTrue(a.evidence_ids or a.tool_refs, f"{a.title} 缺证据")

    def test_intent_routing(self):
        from llm.client import RuleBasedLLM
        self.assertEqual(RuleBasedLLM.classify_intent("帮我把这几张新衣服归类"), "image_archive")
        self.assertEqual(RuleBasedLLM.classify_intent("看看差评里尺码问题"), "comment_insight")
        self.assertEqual(RuleBasedLLM.classify_intent("给我一些选品建议"), "advice")

    def test_trace_recorded(self):
        from orchestration import MultiAgentWorkflow
        st = MultiAgentWorkflow().run("看看评论问题")
        steps = [e.step for e in st.trace]
        self.assertIn("route", steps)
        self.assertIn("tool_call", steps)
        self.assertTrue(st.report)


if __name__ == "__main__":
    unittest.main(verbosity=2)