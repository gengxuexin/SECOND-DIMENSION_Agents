"""运营建议 Agent：把工具输出 + 评论证据合成可执行建议。

可信治理要点（对应 V3）：
- 建议必须带 evidence_ids / tool_refs，pydantic 校验器强制；
- 证据不足的结论降级为"需人工确认"，而不是硬编一个建议；
- 建议置信度由证据量与主题集中度共同决定，不用模型自评。
"""
from __future__ import annotations

from typing import Any

from schemas import Advice, WorkflowState

from .base import BaseAgent


class OperationAdviceAgent(BaseAgent):
    name = "operation_advice"
    tool_names = ["query_comments", "query_products", "analyze_comments", "normalize_tags"]
    system = """你是服装店运营建议 Agent，输出给店主看，必须可执行、可追溯。
硬性要求：
1. 每条建议都要引用具体的评论证据（评论id）或工具结论，不允许输出没有依据的通用建议；
2. 涉及选品/备货的判断要结合商品表里的库存与价格，涉及售后的要结合具体差评主题；
3. 不要杜撰库存数字、销量或评论数量，只能用工具返回的数据。""" + "\n" + BaseAgent.system

    def postprocess(self, state: WorkflowState, observations: list[dict],
                    context: dict) -> dict[str, Any]:
        comments: list[dict] = []
        products: list[dict] = []
        for obs in observations:
            comments.extend(obs.get("comments", []) or [])
            products.extend(obs.get("products", []) or [])
        # 跨节点复用：若本节点未重新取数（编排层复用了上一节点的数据），从上下文取
        if not comments:
            comments = list(context.get("comments") or [])
        if not products:
            products = list(context.get("products") or [])

        advice = self._derive(comments, products, state)
        state.advice = advice
        # 把采集到的商品回传给编排层，供后续节点复用（避免重复查询）
        return {"advice": [a.model_dump() for a in advice],
                "products": products,
                "count": len(advice),
                "need_human": any(a.need_human_confirm for a in advice)}

    # ---------------------------------------------------------------- 规则合成
    def _derive(self, comments: list[dict], products: list[dict],
                state: WorkflowState) -> list[Advice]:
        out: list[Advice] = []
        if not comments:
            return out

        # 以 state.comment_results 为准做品类归属（它记录了真实打分过的评论），
        # 而不是只看 observations 里碰巧出现的评论。
        import store
        index = {c.get("comment_id"): c for c in store.all_comments()}
        by_cat: dict[str, list[str]] = {}
        for cid in (state.comment_results or {}):
            row = index.get(cid) or {}
            if row:
                by_cat.setdefault(row.get("category", "未分类"), []).append(cid)
        # observations 里有但没打分的（例如评论洞察节点未覆盖到的），仅作存在性判断
        seen_cats = {c.get("category") for c in comments}

        low_stock = [p for p in products if p.get("stock", 999) <= 12]
        tool_refs = [f"query_comments#len={len(comments)}"]

        # 1) 差评主题 -> 售后/详情页改进
        for theme in state.themes[:4]:
            if theme.sentiment != "negative":
                continue
            conf = min(0.95, 0.55 + 0.08 * theme.count)
            topic = theme.theme.split("-")[-1]
            out.append(Advice(
                title=f"【{theme.category}】{topic}问题集中，建议优先处理",
                detail=(f"{theme.category}近 {theme.count} 条反馈指向「{topic}」（占该品类主题"
                        f"权重 {theme.share:.2f}）。建议：详情页补充真实尺码对照/实拍细节，"
                        f"并在客服话术中前置说明，降低该类差评与退货。"),
                action_type="售后" if topic in ("质量", "颜色") else "详情页",
                priority="高" if theme.count >= 3 else "中",
                confidence=round(conf, 4),
                tool_refs=tool_refs,
                evidence_ids=theme.evidence_ids,
                need_human_confirm=theme.count < 2,
            ))

        # 2) 库存 -> 备货建议（品类评论证据必须真实存在，否则只给库存事实不给判断）
        for p in low_stock[:3]:
            cat = p.get("category", "")
            evidence = list(dict.fromkeys(by_cat.get(cat, [])))[:5]
            has_signal = bool(evidence)
            detail = (f"{cat} 品类下该 SKU 库存 {p.get('stock')} 件（定价 {p.get('price')}）。")
            if has_signal:
                detail += (f"该品类本期已分析 {len(by_cat[cat])} 条评论，"
                           f"建议结合上面评论反馈判断是补码还是改版：若差评集中在尺码，优先补主推码；"
                           f"若集中在质量，先修版再补货，避免压货。")
            else:
                detail += "该品类本期未取到评论样本，补货决策前建议先补齐评论数据再判断。"
            out.append(Advice(
                title=f"【{p.get('name', '')}】库存仅 {p.get('stock', 0)} 件，需确认补货",
                detail=detail,
                action_type="备货",
                priority="高" if has_signal else "中",
                confidence=round(min(0.9, 0.6 + 0.05 * len(evidence)), 4),
                tool_refs=[f"query_products#sku={p.get('sku_id', '')}"] + tool_refs,
                evidence_ids=evidence,
                need_human_confirm=not has_signal,
            ))

        # 3) 正面反馈 -> 选品/陈列放大
        pos_by_cat: dict[str, list[str]] = {}
        for t in state.themes:
            if t.sentiment == "positive":
                pos_by_cat.setdefault(t.category, []).extend(t.evidence_ids)
        for cat, ids in sorted(pos_by_cat.items(),
                               key=lambda kv: -len(kv[1]))[:2]:
            uniq = list(dict.fromkeys(ids))[:6]
            topics = {t.theme.split("-")[-1] for t in state.themes
                      if t.category == cat and t.sentiment == "positive"}
            out.append(Advice(
                title=f"【{cat}】正面反馈集中，可加大陈列与备货权重",
                detail=(f"{cat} 品类获得 {len(uniq)} 条正面主题反馈"
                        f"（{'、'.join(sorted(topics))}）。"
                        f"建议：线上主图强化该卖点，线下陈列面扩大，补货比例上调。"),
                action_type="选品",
                priority="中",
                confidence=round(min(0.9, 0.55 + 0.06 * len(uniq)), 4),
                tool_refs=tool_refs,
                evidence_ids=uniq,
            ))

        # 证据不足时，宁可少给也不硬编
        out = [a for a in out if a.evidence_ids or (a.tool_refs and a.confidence >= 0.6)]
        out.sort(key=lambda a: ({"高": 0, "中": 1, "低": 2}[a.priority], -a.confidence))
        return out