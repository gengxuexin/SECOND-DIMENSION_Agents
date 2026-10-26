"""评论洞察 Agent：评论 -> 逐条情感/主题打分 -> 按商品/品类聚合 -> 主题归纳（带证据）。"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from schemas import CommentScores, CommentTheme, WorkflowState

from .base import BaseAgent


class CommentInsightAgent(BaseAgent):
    name = "comment_insight"
    tool_names = ["query_comments", "analyze_comments", "query_products"]
    system = """你是网店评论洞察 Agent。
流程：先用 query_comments 取评论（scope=all 或指定品类/商品），再对评论批量调用 analyze_comments 拿情感与主题打分。
然后按商品/品类聚合，把同一主题归纳成组，每个主题必须带上支撑它的评论 id 作为证据。
禁止输出没有评论证据的主题；不要凭印象总结店主店铺情况。""" + "\n" + BaseAgent.system

    def postprocess(self, state: WorkflowState, observations: list[dict],
                    context: dict) -> dict[str, Any]:
        # 建立 comment_id -> 评论正文，保证打分结果与评论严格对齐（不依赖顺序假设）
        comments: list[dict] = []
        products: list[dict] = []
        for obs in observations:
            for c in obs.get("comments", []) or []:
                comments.append(c)
            products.extend(obs.get("products", []) or [])

        # 建立 comment_id -> 评论正文，保证打分结果与评论严格对齐（不依赖顺序假设）
        by_id: dict[str, dict] = {}
        for c in comments:
            if c.get("comment_id"):
                by_id[c["comment_id"]] = c
        # 正文 -> comment_id 反查：工具返回的 results 与入参顺序一致，但入参只有正文
        id_by_text = {(c.get("text") or "").strip(): cid for cid, c in by_id.items()}

        # 记录 analyze_comments 的入参顺序，用于与返回的 results 按位对齐
        order: list[str] = list(getattr(self, "_analyze_order", []) or [])
        scored: list[tuple[dict, CommentScores]] = []
        for obs in observations:
            for i, res in enumerate(obs.get("results", []) or []):
                if not res.get("scores"):
                    continue
                cid = order[i] if i < len(order) else f"pos-{len(scored)}"
                cid = id_by_text.get(cid.strip(), cid)
                row = by_id.get(cid, {"comment_id": cid, "text": "", "category": ""})
                scored.append((row, CommentScores(**res["scores"])))

        state.comment_results = {c["comment_id"]: s for c, s in scored}
        themes = self._aggregate(scored)
        state.themes = themes

        pos = sum(1 for _, s in scored if s.sentiment == "positive")
        neg = sum(1 for _, s in scored if s.sentiment == "negative")
        neu = len(scored) - pos - neg
        dist = {"positive": pos, "negative": neg, "neutral": neu}
        total = max(len(scored), 1)

        return {
            "analyzed": len(scored),
            "sentiment_dist": dist,
            "negative_rate": round(neg / total, 4),
            "themes": [t.model_dump() for t in themes[:10]],
            "by_category": self._by_category(scored),
            "comments": [dict(c, text=c.get("text", "")) for c, _ in scored],
            "products": products,
            "need_human": False,
        }

    # ---------------------------------------------------------------- 聚合
    def _aggregate(self, scored: list[tuple[dict, CommentScores]]) -> list[CommentTheme]:
        buckets: dict[tuple[str, str], list[tuple[dict, float]]] = defaultdict(list)
        for c, s in scored:
            cat = c.get("category", "") or "未分类"
            for topic, conf in (s.topics or {}).items():
                buckets[(cat, topic)].append((c, conf))

        themes: list[CommentTheme] = []
        for (cat, topic), items in buckets.items():
            if not items:
                continue
            ids = [c["comment_id"] for c, _ in items]
            avg_conf = sum(x for _, x in items) / len(items)
            pol = self._polarity(items)
            sample = items[0][0].get("text", "")[:60]
            themes.append(CommentTheme(
                theme=f"{cat}-{topic}", category=cat, count=len(items),
                sentiment=pol, share=round(avg_conf, 4),
                evidence_ids=ids[:8], sample_excerpt=sample))
        themes.sort(key=lambda t: (-t.count, -t.share))
        return themes

    def _polarity(self, items: list[tuple[dict, float]]) -> str:
        texts = " ".join(c.get("text", "") for c, _ in items)
        neg = sum(texts.count(w) for w in ("差", "烂", "偏小", "偏大", "起球", "色差",
                                          "开线", "掉色", "失望", "退货", "退款"))
        pos = sum(texts.count(w) for w in ("好评", "满意", "喜欢", "舒服", "不错", "值", "回购"))
        if neg > pos:
            return "negative"
        if pos > neg:
            return "positive"
        return "neutral"

    def _by_category(self, scored: list[tuple[dict, CommentScores]]) -> dict[str, Any]:
        agg: dict[str, Counter] = defaultdict(Counter)
        for c, s in scored:
            agg[c.get("category", "未分类")][s.sentiment] += 1
        return {k: dict(v) for k, v in agg.items()}