"""V2 多 Agent 业务闭环：显式状态机编排。

状态流转：
    PENDING ──route──> ARCHIVING ──> INSIGHT ──> ADVISING ──> REPORT ──> DONE
                          │            │           │
                          └─ 低置信 ────┴───────────┴─> HUMAN_REVIEW ──> DONE
                          │            │           │
                          └────────────┴───────────┴─> FAILED（工具不可达等）

状态机本身不含任何业务判断，只负责：
1. 意图路由决定需要经过哪些节点；
2. 节点间的数据依赖（图像标签 -> 评论洞察的聚合维度 -> 建议的证据来源）；
3. 异常与人工介入的旁路。
这样单节点可独立单测，链路可整体回归。
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from agents import CommentInsightAgent, ImageArchiveAgent, OperationAdviceAgent
from llm.client import RuleBasedLLM, get_llm
from schemas import TaskStatus, WorkflowState

NODE_ARCHIVE = "image_archive"
NODE_INSIGHT = "comment_insight"
NODE_ADVICE = "operation_advice"


def _harvest(state: WorkflowState, node_out: dict[str, Any],
             harvested: dict[str, Any]) -> None:
    """把本节点采集到的数据放进共享池，供后续节点复用。

    这是 V2"多 Agent 共享状态"的直接收益：评论洞察取过一次评论，
    运营建议节点就不必重复调用 query_comments，省一次网络往返与重复推理。
    """
    import store
    # 评论：优先用节点显式回传的（带打分上下文），否则从 state 的打分结果反查
    if node_out.get("comments"):
        harvested["comments"] = node_out["comments"]
    elif state.comment_results and not harvested["comments"]:
        index = {c.get("comment_id"): c for c in store.all_comments()}
        harvested["comments"] = [index[cid] for cid in state.comment_results
                                 if cid in index]
    # 商品：任一节点回传即可
    if node_out.get("products") and not harvested["products"]:
        harvested["products"] = node_out["products"]


class MultiAgentWorkflow:
    """多 Agent 编排器。一个实例管理一次任务的生命周期。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or get_llm()
        self.archive = ImageArchiveAgent(self.llm)
        self.insight = CommentInsightAgent(self.llm)
        self.advice = OperationAdviceAgent(self.llm)

    # ---------------------------------------------------------------- 路由
    def route(self, state: WorkflowState) -> list[str]:
        """意图 -> 执行节点序列。规则路由稳定可测，大模型仅在文本歧义时辅助。"""
        text = state.user_query
        if isinstance(self.llm, RuleBasedLLM):
            intent = RuleBasedLLM.classify_intent(text)
        else:
            intent = self._llm_route(state)
        state.intent = intent
        state.log("route", "system", f"intent={intent}")

        nodes = [NODE_ARCHIVE] if state.image_paths else []
        if intent in ("comment_insight", "advice", "unknown") or not nodes:
            nodes.append(NODE_INSIGHT)
        if intent in ("advice", "unknown") or (state.themes and intent == "comment_insight"):
            nodes.append(NODE_ADVICE)
        return nodes

    def _llm_route(self, state: WorkflowState) -> str:
        prompt = (
            "判断用户请求属于哪个场景，只输出一个词。\n"
            "场景定义：\n"
            "- image_archive：上传/拍照/归档/识别衣服图片、做品类款式归类\n"
            "- comment_insight：评论分析、差评/好评分布、尺码/色差/质量问题发现\n"
            "- advice：基于以上信息要选品/备货/售后/运营建议\n"
            f"用户请求：{state.user_query}\n输出："
        )
        try:
            resp = self.llm.chat_with_tools(
                [{"role": "user", "content": prompt}], [], "你是意图分类器，只输出一个词。", max_rounds=1)
            text = (resp.get("content") or "").strip().lower()
            if any(k in text for k in ("image_archive", "comment_insight", "advice")):
                for k in ("image_archive", "comment_insight", "advice"):
                    if k in text:
                        return k
        except Exception as e:
            state.errors.append(f"intent 路由失败，降级规则路由: {e}")
        return RuleBasedLLM.classify_intent(state.user_query)

    # ---------------------------------------------------------------- 执行
    def run(self, user_query: str, image_paths: list[str] | None = None,
            comment_scope: str = "all", comments: list[str] | None = None,
            task_id: str | None = None) -> WorkflowState:
        state = WorkflowState(task_id=task_id or f"T-{uuid.uuid4().hex[:8]}",
                              user_query=user_query, image_paths=image_paths or [],
                              comment_scope=comment_scope)
        context = {"image_paths": state.image_paths, "comment_scope": comment_scope,
                   "comments": comments or []}
        # 跨 Agent 复用：后续节点不必重复取同一批数据（V2 共享状态的直接收益）。
        # 评论洞察已经拿到评论 -> 运营建议节点直接复用，不再重复调用 query_comments。
        harvested: dict[str, Any] = {"comments": [], "products": []}
        t0 = time.perf_counter()

        try:
            nodes = self.route(state)
        except Exception as e:
            state.errors.append(f"路由异常: {e}")
            state.status = TaskStatus.FAILED
            return state

        state.log("plan", "system", " -> ".join(nodes), {"nodes": nodes})
        outputs: dict[str, Any] = {}
        gated: list[str] = []

        for node in nodes:
            handler: Callable[..., dict] = {
                NODE_ARCHIVE: self.archive.run,
                NODE_INSIGHT: self.insight.run,
                NODE_ADVICE: self.advice.run,
            }[node]
            try:
                # 把上一节点采集的数据注入当前节点上下文，实现跨节点复用
                node_ctx = {**context, **{k: v for k, v in harvested.items() if v}}
                outputs[node] = handler(state, node_ctx, max_rounds=4)
                _harvest(state, outputs[node], harvested)
            except Exception as e:
                state.errors.append(f"{node} 执行失败: {type(e).__name__}: {e}")
                state.log("node_error", "system", f"{node}: {e}")
                continue

            # 置信度旁路：低置信项不自动采信，转人工确认。
            # 注意这里"标记而不中断"：图像归档的低置信标签是结论待复核，
            # 但后续评论洞察/运营建议依赖的是评论证据而非图像标签，
            # 一并中断会让店主白等，也违背"人工保留确认权"的本意（确认权不等于必须阻塞）。
            if outputs[node].get("need_human"):
                reason = (outputs[node].get("low_conf_items")
                          or [k for k, v in outputs[node].items() if v is True] or ["低置信结果"])
                gated.append(f"{node}: {reason}")
                state.log("human_gate", "system",
                          f"{node} 存在待人工确认项 {reason}，标记后继续后续节点",
                          {"gate": "low_confidence", "node": node})

        state.status = TaskStatus.NEEDS_HUMAN if gated else TaskStatus.DONE
        state.report = self._report(state, outputs)
        state.log("finish", "system", f"耗时 {round((time.perf_counter() - t0) * 1000, 2)}ms",
                  {"gated": gated})
        return state

    # ---------------------------------------------------------------- 报告
    def _report(self, state: WorkflowState, outputs: dict[str, Any],
                partial: bool = False) -> str:
        L: list[str] = [f"# 服装店智能运营报告（task={state.task_id}）", ""]
        L.append(f"- 意图路由：`{state.intent}`　编排后端：`{self.llm.name}`"
                 f"　执行节点：{'、'.join(outputs) or '无'}　状态：`{state.status.value}`")
        if state.low_conf_items:
            L.append(f"- ⚠️ 待人工确认（不自动采信）：{', '.join(state.low_conf_items)}")
        if state.errors:
            L.append(f"- ⚠️ 异常：{'; '.join(state.errors[:4])}")
        L.append("")

        if NODE_ARCHIVE in outputs:
            o = outputs[NODE_ARCHIVE]
            L.append("## 一、图像归档")
            for t in o.get("labels", []):
                flag = " ⚠️待人工确认" if t.get("need_confirm") else ""
                L.append(f"- `{t['raw']}` → **{t['normalized']}**"
                         f"（置信度 {t['conf']:.2f}，来源 {t['source']}）{flag}")
            if o.get("low_conf_items"):
                L.append(f"- 低置信项：{o['low_conf_items']}（已转人工，不自动采信）")
            L.append("")

        if NODE_INSIGHT in outputs:
            o = outputs[NODE_INSIGHT]
            d = o.get("sentiment_dist", {})
            L.append("## 二、评论洞察")
            L.append(f"- 分析 {o.get('analyzed', 0)} 条：正面 {d.get('positive', 0)} / "
                     f"负面 {d.get('negative', 0)} / 中性 {d.get('neutral', 0)}"
                     f"（负面占比 {o.get('negative_rate', 0):.1%}）")
            L.append("")
            L.append("| 主题 | 品类 | 条数 | 倾向 | 证据评论 |")
            L.append("|---|---|---|---|---|")
            for t in o.get("themes", [])[:8]:
                topic = t["theme"].split("-")[-1]
                L.append(f"| {topic} | {t['category']} | {t['count']} | {t['sentiment']} | "
                         f"{', '.join(t['evidence_ids'][:5])} |")
            L.append("")

        if NODE_ADVICE in outputs:
            L.append("## 三、运营建议")
            for a in outputs[NODE_ADVICE].get("advice", [])[:6]:
                L.append(f"### {a['title']}（{a['priority']}优先 / 置信度 {a['confidence']:.2f}）")
                L.append(a["detail"])
                L.append(f"- 引用工具：{a['tool_refs']}")
                L.append(f"- 证据评论：{', '.join(a['evidence_ids']) or '（无评论证据）'}")
                L.append("")

        if partial:
            L.append("> 本次链路在人工确认点暂停，以上为已确认部分。")
        return "\n".join(L)


def auto_confirm(state: WorkflowState, node_out: dict[str, Any]) -> bool:
    """默认策略：等待人工确认。此处预留 CLI/服务端的自动确认开关（如配置 AUTO_CONFIRM=1）。"""
    import os
    if os.getenv("AUTO_CONFIRM", "0") != "1":
        return False
    import store
    for raw in node_out.get("low_conf_items", []):
        store.append_human_sample({"kind": "tag", "raw": raw, "confirmed": raw,
                                   "task_id": state.task_id, "by": "auto_confirm"})
    return True


__all__ = ["MultiAgentWorkflow", "NODE_ARCHIVE", "NODE_INSIGHT", "NODE_ADVICE"]