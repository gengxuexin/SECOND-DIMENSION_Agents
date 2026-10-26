"""V3 可信治理：置信度分流、人工确认队列、人工修正回流、证据校验。"""
from __future__ import annotations

import json
import uuid
from typing import Any

import store
from schemas import Advice, CommentTheme, NormalizedTag, WorkflowState


# ---------------------------------------------------------------- 置信度分流
def triage(state: WorkflowState) -> dict[str, Any]:
    """把整条链路的结果按置信度分层，决定哪些可直接采信、哪些必须人工过。"""
    auto_ok: list[str] = []
    need_human: list[str] = []

    for t in state.image_labels:
        key = f"tag:{t.raw}"
        (need_human if t.need_confirm or t.conf < 0.9 else auto_ok).append(key)

    for cid, s in (state.comment_results or {}).items():
        key = f"comment:{cid}"
        (need_human if s.sentiment_conf < 0.6 else auto_ok).append(key)

    for th in state.themes:
        key = f"theme:{th.theme}"
        # 主题只有 1 条证据时，归纳风险高，必须人工复核
        (need_human if th.count < 2 or len(th.evidence_ids) < 2 else auto_ok).append(key)

    for a in state.advice:
        key = f"advice:{a.title}"
        (need_human if a.need_human_confirm or a.confidence < 0.6 else auto_ok).append(key)

    return {"auto_ok": auto_ok, "need_human": need_human,
            "auto_rate": round(len(auto_ok) / max(len(auto_ok) + len(need_human), 1), 4)}


# ---------------------------------------------------------------- 人工确认队列
class ReviewQueue:
    """低置信结果的人工确认队列。确认/修正结果回流样本库，供评测 few-shot 复用。"""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, kind: str, payload: dict[str, Any], reason: str,
            task_id: str) -> dict[str, Any]:
        item = {"item_id": f"H-{uuid.uuid4().hex[:6]}", "kind": kind, "payload": payload,
                "reason": reason, "task_id": task_id, "status": "pending", "decision": None}
        self.items.append(item)
        return item

    def from_state(self, state: WorkflowState) -> list[dict[str, Any]]:
        for t in state.image_labels:
            if t.need_confirm:
                self.add("tag_normalize", t.model_dump(),
                         f"标签归一置信度 {t.conf:.2f} 低于阈值", state.task_id)
        for th in state.themes:
            if th.count < 2:
                self.add("comment_theme", th.model_dump(),
                         f"主题仅 {th.count} 条证据，归纳风险高", state.task_id)
        for a in state.advice:
            if a.need_human_confirm or not a.evidence_ids:
                self.add("advice", a.model_dump(),
                         "建议证据不足或置信度偏低", state.task_id)
        return self.items

    def resolve(self, item_id: str, decision: str, corrected_value: Any = None,
                reviewer: str = "owner") -> dict[str, Any] | None:
        for it in self.items:
            if it["item_id"] == item_id:
                it["status"] = "resolved"
                it["decision"] = decision
                it["corrected_value"] = corrected_value
                it["reviewer"] = reviewer
                store.append_human_sample({"kind": it["kind"], "task_id": it["task_id"],
                                           "original": it["payload"], "decision": decision,
                                           "corrected_value": corrected_value,
                                           "reason": it["reason"], "by": reviewer})
                return it
        return None


# ---------------------------------------------------------------- 证据校验
def validate_evidence(state: WorkflowState) -> dict[str, Any]:
    """校验每条建议/主题的证据是否真实存在于本轮评论集合中（防幻觉闭环校验）。"""
    valid_ids = set((state.comment_results or {}).keys())
    report = {"advice_checked": 0, "advice_ok": 0, "bad_evidence": [], "orphan_themes": []}

    for a in state.advice:
        report["advice_checked"] += 1
        ev = [i for i in a.evidence_ids if i in valid_ids]
        if ev:
            report["advice_ok"] += 1
        elif not a.tool_refs:
            report["bad_evidence"].append(a.title)
        a.evidence_ids = ev

    for th in state.themes:
        ev = [i for i in th.evidence_ids if i in valid_ids]
        if not ev:
            report["orphan_themes"].append(th.theme)
        th.evidence_ids = ev

    report["hallucination_free"] = not report["bad_evidence"] and not report["orphan_themes"]
    return report


__all__ = ["triage", "ReviewQueue", "validate_evidence"]