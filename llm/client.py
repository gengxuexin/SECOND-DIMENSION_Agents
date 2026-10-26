"""LLM 编排层客户端。

职责边界（对应项目"LLM 负责理解与编排，模型负责判定"）：
- 只做：意图理解、工具选择与参数生成、标签归一候选、评论主题措辞、建议文本生成；
- 不做：自己给出品类判定、情感判定（必须来自 tools/ 里的确定性服务）。

两种运行模式：
- 配了 LLM_API_KEY：走 OpenAI 兼容接口，temperature=0，开启 tools 严格模式；
- 未配置：降级为 RuleBasedLLM（关键词/正则路由 + 模板生成），保证离线可跑、可做回归测试。
两种模式都受 max_rounds 与严格工具校验约束，行为差异在评测报告中显式标注。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

import config
from schemas import ToolCallRecord


class BaseLLM:
    name = "base"

    def chat_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                        system: str, max_rounds: int = 4) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    """DeepSeek / OpenAI / 通义千问 等 OpenAI 兼容端点通用实现。"""

    def __init__(self) -> None:
        self.name = f"openai-compatible:{config.LLM_MODEL}"

    def chat_with_tools(self, messages, tools, system, max_rounds=4):
        history = [{"role": "system", "content": system}] + list(messages)
        payload = {
            "model": config.LLM_MODEL,
            "messages": history,
            "temperature": config.LLM_TEMPERATURE,
            "tools": tools,
            "tool_choice": "auto",
        }
        raw = self._post(payload)
        msg = raw["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return {"tool_calls": [], "content": msg.get("content", ""), "raw": raw}
        return {
            "tool_calls": [
                {
                    "call_id": tc.get("id", f"call_{i}"),
                    "name": tc["function"]["name"],
                    "arguments": _safe_json(tc["function"].get("arguments", "{}")),
                }
                for i, tc in enumerate(tcs)
            ],
            "content": msg.get("content", ""),
            "raw": raw,
        }

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {config.LLM_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))


class RuleBasedLLM(BaseLLM):
    """离线降级编排器：关键词路由 + 模板生成。

    它不替代大模型，只保证"没配 Key 也能演示与跑回归测试"，并且在 trace 中明确标注
    backend=rule_based，避免把降级结果误当成真实大模型能力。
    """

    name = "rule-based-offline"

    INTENT_RULES: list[tuple[str, list[str]]] = [
        ("image_archive", ["图片", "拍照", "归档", "归类", "分类", "这件", "上传", "识别", "款式"]),
        ("comment_insight", ["评论", "差评", "好评", "反馈", "口碑", "退货", "尺码", "起球", "色差"]),
        ("advice", ["建议", "选品", "备货", "怎么弄", "运营", "改进", "怎么办", "帮我出"]),
    ]

    def chat_with_tools(self, messages, tools, system, max_rounds=4):
        """离线编排器必须"看对话历史"推进，否则会重复调用同一工具。

        真实大模型天然具备这个行为；降级实现需要显式地从已发生的 tool 消息里
        判断进度：已取到评论 -> 该打分；已打分 -> 该收敛结束。
        """
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content", "")
                break

        extra = self._context_of(messages)
        t0 = time.perf_counter()
        intent = self.classify_intent(user)
        calls = self._plan_calls(intent, user, extra, tools)
        return {"tool_calls": calls, "content": "", "raw": {"backend": "rule_based"},
                "cost_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @staticmethod
    def _context_of(messages: list[dict[str, Any]]) -> dict[str, Any]:
        """从 system 槽位的 JSON 上下文读取任务参数与已取回数据。

        Agent 每轮都会把 live 快照写进 system 槽位（见 BaseAgent._absorb），
        所以这里读到的就是最新进度，无需解析历史 tool 消息。
        """
        for m in messages:
            if m.get("role") == "system":
                try:
                    obj = json.loads(m.get("content", "{}"))
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        return {}

    @classmethod
    def classify_intent(cls, text: str) -> str:
        scores = {name: sum(1 for w in words if w in text) for name, words in cls.INTENT_RULES}
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else "unknown"

    def _plan_calls(self, intent: str, text: str, extra: dict,
                    tools: list[dict]) -> list[dict[str, Any]]:
        names = {t["function"]["name"] for t in tools}
        called = set(extra.get("called_tools") or [])
        comments = extra.get("comments") or []
        products = extra.get("products") or []
        scope = extra.get("comment_scope") or "all"
        calls: list[dict[str, Any]] = []

        def add(name: str, args: dict):
            if name in names:
                calls.append({"call_id": f"rb_{len(calls) + 1}", "name": name,
                              "arguments": args})

        # ---------- 图像归档：按路径逐张分类（已分类过的不重复调用）
        if intent == "image_archive" or extra.get("image_paths"):
            done_paths = set(extra.get("classified_paths") or [])
            pending = [p for p in (extra.get("image_paths") or []) if p not in done_paths]
            if pending:
                for p in pending:
                    add("classify_image", {"image_path": p, "top_k": 3})
            elif extra.get("image_predictions") and "normalize_tags" not in called:
                raws = list(dict.fromkeys(
                    p["label"] for p in extra["image_predictions"] if p.get("label")))
                if raws:
                    add("normalize_tags", {"raw_tags": raws[:20]})

        # ---------- 评论洞察：先取评论，再对正文打分（取过就不再取）
        if intent in ("comment_insight", "advice", "unknown"):
            have = bool(comments)
            if have and "analyze_comments" not in called:
                texts = list(dict.fromkeys(c.get("text", "") for c in comments if c.get("text")))
                if texts:
                    add("analyze_comments", {"comments": texts[:12], "top_n": 3})
            elif not have and "query_comments" not in called:
                add("query_comments", {"scope": scope, "limit": 12})

        # ---------- 商品上下文（运营建议需要库存/价格）
        if intent == "advice" and not products and "query_products" not in called:
            add("query_products", {"limit": 20})
        return calls


def _safe_json(s: str) -> dict:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        return {}


def get_llm() -> BaseLLM:
    if config.LLM_API_KEY:
        return OpenAICompatibleLLM()
    return RuleBasedLLM()


__all__ = ["BaseLLM", "OpenAICompatibleLLM", "RuleBasedLLM", "get_llm", "ToolCallRecord"]