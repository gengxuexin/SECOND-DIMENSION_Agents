"""V3 评测与回归：小规模人工标注评测集。

四类用例：
1. intent：意图路由准确率
2. tag_norm：标签归一准确率 + 待确认率
3. comment_topic：评论主题召回 / 情感准确率（对照评论库 gold 标注）
4. advice_evidence：建议证据闭环率（是否幻觉）

跑分命令：python run.py eval
"""
from __future__ import annotations

import json
import time
from typing import Any

import config
import store
from llm.client import RuleBasedLLM, get_llm
from tools.registry import invoke


def _load() -> list[dict[str, Any]]:
    if not config.EVAL_SET.exists():
        import data_seed
        data_seed.write_all()
    return [json.loads(l) for l in config.EVAL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_eval() -> dict[str, Any]:
    rows = _load()
    llm = get_llm()
    per_kind: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []

    def bucket(kind: str) -> dict[str, Any]:
        return per_kind.setdefault(kind, {"n": 0, "pass": 0, "fail": 0})

    for r in rows:
        kind = r["kind"]
        t0 = time.perf_counter()
        ok, note = _case(r, llm)
        cost = round((time.perf_counter() - t0) * 1000, 2)
        b = bucket(kind)
        b["n"] += 1
        b["pass" if ok else "fail"] += 1
        b.setdefault("cost_ms", 0.0)
        b["cost_ms"] = round(b["cost_ms"] + cost, 2)
        details.append({"id": r["id"], "kind": kind, "input": r["input"],
                        "pass": ok, "note": note, "cost_ms": cost})

    for b in per_kind.values():
        b["accuracy"] = round(b["pass"] / b["n"], 4)
        b["avg_cost_ms"] = round(b["cost_ms"] / b["n"], 2)

    total_pass = sum(b["pass"] for b in per_kind.values())
    total_n = sum(b["n"] for b in per_kind.values())
    return {"llm_backend": llm.name, "overall": round(total_pass / max(total_n, 1), 4),
            "total": total_n, "passed": total_pass,
            "by_kind": per_kind, "details": details,
            "human_samples": len(store.load_human_samples())}


def _case(r: dict[str, Any], llm: Any) -> tuple[bool, str]:
    kind, inp = r["kind"], r["input"]
    try:
        if kind == "intent":
            got = RuleBasedLLM.classify_intent(inp) if isinstance(llm, RuleBasedLLM) \
                else _llm_intent(llm, inp)
            exp = r["expect"]["intent"]
            return got == exp, f"intent={got} expect={exp}"

        if kind == "tag_norm":
            nt = store.normalize_tag(inp)
            exp = r["expect"]
            ok = nt.normalized == exp["normalized"] and nt.need_confirm == exp["need_confirm"]
            return ok, f"{inp} -> {nt.normalized}({nt.conf:.2f},need={nt.need_confirm})"

        if kind == "comment_topic":
            out, _ = invoke("analyze_comments", {"comments": [inp], "top_n": 4})
            scores = (out.get("results") or [{}])[0].get("scores") or {}
            exp = r["expect"]
            got_topics = list((scores.get("topics") or {}).keys())
            if "topics_include" in exp:
                miss = [t for t in exp["topics_include"] if t not in got_topics]
                sent_ok = (not exp.get("sentiment")) or scores.get("sentiment") == exp["sentiment"]
                return (not miss and sent_ok), f"topics={got_topics} sent={scores.get('sentiment')}"
            return scores.get("sentiment") == exp["sentiment"], \
                f"sentiment={scores.get('sentiment')}"

        if kind == "advice_evidence":
            from orchestration import MultiAgentWorkflow
            st = MultiAgentWorkflow(llm).run("根据评论和库存给我一些选品建议", comment_scope="all")
            from governance.trust import validate_evidence
            vr = validate_evidence(st)
            exp = r["expect"]
            n = len(st.advice)
            ok = n >= exp.get("min_advice", 1) and \
                (not exp.get("require_evidence") or vr["hallucination_free"])
            return ok, f"advice={n} hallucination_free={vr['hallucination_free']}"

        if kind == "tool_args":
            exp = r["expect"]
            from tools.registry import sanitize_args
            dirty = {"comments": ["x"], **{k: "malicious" for k in exp["forbidden_args"]}}
            clean = sanitize_args(exp["tool"], dirty)
            leaked = [k for k in exp["forbidden_args"] if k in clean]
            return not leaked, f"cleaned={list(clean)} leaked={leaked}"

        return True, "unknown kind, skipped"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _llm_intent(llm: Any, text: str) -> str:
    resp = llm.chat_with_tools(
        [{"role": "user", "content":
          f"场景定义：image_archive=图片归类；comment_insight=评论分析；advice=运营建议。\n"
          f"请求：{text}\n输出场景名："}],
        [], "你是意图分类器。", max_rounds=1)
    out = (resp.get("content") or "").strip().lower()
    for k in ("image_archive", "comment_insight", "advice"):
        if k in out:
            return k
    return "unknown"


def print_report(res: dict[str, Any]) -> None:
    print("=" * 68)
    print(f"评测报告  编排后端={res['llm_backend']}  用例={res['total']}")
    print("=" * 68)
    print(f"{'用例类型':<18}{'通过':>6}{'总数':>6}{'准确率':>10}{'均耗时ms':>10}")
    for kind, b in res["by_kind"].items():
        print(f"{kind:<18}{b['pass']:>6}{b['n']:>6}{b['accuracy']:>10.2%}{b['avg_cost_ms']:>10.2f}")
    print("-" * 68)
    print(f"总体准确率：{res['overall']:.2%}  ({res['passed']}/{res['total']})"
          f"  人工回流样本：{res['human_samples']} 条")
    print("=" * 68)
    fails = [d for d in res["details"] if not d["pass"]]
    if fails:
        print("未通过用例：")
        for d in fails:
            print(f"  [{d['id']}] {d['input']} -> {d['note']}")
    else:
        print("全部用例通过 ✓")


__all__ = ["run_eval", "print_report"]