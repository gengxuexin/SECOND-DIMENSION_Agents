"""图像归档 Agent：图片 -> 品类/款式标签 -> 标签库归一 -> 低置信转人工。"""
from __future__ import annotations

from typing import Any

from schemas import NormalizedTag, WorkflowState

from .base import BaseAgent


class ImageArchiveAgent(BaseAgent):
    name = "image_archive"
    tool_names = ["classify_image", "normalize_tags", "query_products"]
    system = """你是服装门店的图像归档 Agent。
流程：先对每张图片调用 classify_image 得到 Top-K 标签，把 Top-1 标签交给 normalize_tags 归一到标准标签库。
对置信度低于阈值的图片，必须标记为待人工确认，不要自行拍板。
不要输出图片之外不存在的商品信息。""" + "\n" + BaseAgent.system

    def postprocess(self, state: WorkflowState, observations: list[dict],
                    context: dict) -> dict[str, Any]:
        import config

        labels: list[NormalizedTag] = []
        low_conf: list[str] = []
        raw_top: list[dict] = []

        for obs in observations:
            if "predictions" in obs and obs.get("ok", True):
                preds = obs.get("predictions") or []
                if not preds:
                    continue
                raw_top.append(preds[0])
                # 标签归一优先用工具返回；未归一时在本地标签库兜底（同一套规则）
                norm_map = {}
                for n in observations:
                    for item in n.get("normalized", []):
                        norm_map[item["raw"]] = item
                item = norm_map.get(preds[0]["label"])
                if item:
                    nt = NormalizedTag(**item)
                else:
                    import store
                    nt = store.normalize_tag(preds[0]["label"])
                labels.append(nt)
                # 三条转人工的条件（任一满足即不自动采信）：
                # 1) 标签归一本身置信度不足
                # 2) 分类置信度低于全局阈值
                # 3) 多后端交叉验证结论冲突（工具层已标记 need_confirm）
                conflict = preds[0].get("need_confirm")
                if nt.need_confirm or preds[0]["score"] < config.IMAGE_LOW_CONF or conflict:
                    low_conf.append(preds[0]["label"])

        state.image_labels = labels
        state.low_conf_items = low_conf
        return {"labels": [l.model_dump() for l in labels],
                "low_conf_items": low_conf,
                "raw_top1": raw_top,
                "archived": len(labels),
                "need_human": bool(low_conf)}