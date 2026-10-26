"""V1 单 Agent 工具化回归：验证 Function Calling 工具可用、参数可校验。

这一层不参与 V2 业务闭环，作用是：
1. 证明"两个模型服务能被业务直接调用"（V1 的核心命题）；
2. 为工具层建立独立的可回归契约，作为 V2/V3 的地基。
"""
from __future__ import annotations

from typing import Any

from llm.client import RuleBasedLLM, get_llm
from schemas import WorkflowState
from tools.registry import TOOL_SPECS, invoke


def single_agent_toolcall(user_query: str, context: dict[str, Any] | None = None,
                          llm: Any | None = None) -> dict[str, Any]:
    """用 Function Calling 把工具跑一轮，返回全部调用与结果。"""
    import json
    from agents.base import _as_openai_calls, _json

    llm = llm or get_llm()
    context = context or {}
    state = WorkflowState(task_id="v1-single-agent", user_query=user_query)
    tools = TOOL_SPECS

    messages = [{"role": "system", "content": _json(context)},
                {"role": "user", "content": user_query}]
    results: list[dict[str, Any]] = []

    if isinstance(llm, RuleBasedLLM):
        # 走一遍完整的多轮循环，与 V2 的 Agent 执行路径保持一致
        calls: list[dict[str, Any]] = []
        live = dict(context)
        for _ in range(3):
            step = llm.chat_with_tools(
                [{"role": "system", "content": _json(live)},
                 {"role": "user", "content": user_query}],
                tools, "你是工具调用助手。", max_rounds=1).get("tool_calls") or []
            if not step:
                break
            calls.extend(step)
            for c in step:
                out, rec = invoke(c["name"], c.get("arguments") or {})
                _absorb_into(live, c["name"], out)
    else:
        resp = llm.chat_with_tools(messages, tools, "你是工具调用助手。", max_rounds=1)
        calls = resp.get("tool_calls") or []

    for c in calls:
        out, rec = invoke(c["name"], c.get("arguments") or {})
        results.append({"call_id": c["call_id"], "name": c["name"],
                        "args": rec.arguments, "ok": rec.ok,
                        "cost_ms": rec.cost_ms, "summary": rec.result_summary,
                        "result": out})
        state.log("tool_call", "tool", f"{c['name']} -> {rec.result_summary}",
                  {"args": rec.arguments}, cost_ms=rec.cost_ms)
    return {"llm": llm.name, "calls": results, "trace": [t.model_dump() for t in state.trace]}


def _absorb_into(live: dict[str, Any], tool_name: str, out: dict) -> None:
    """与 BaseAgent 共用同一套上下文回灌逻辑，保证 V1/V2 行为一致。"""
    from agents.base import absorb_tool_result
    absorb_tool_result(live, tool_name, out)


__all__ = ["single_agent_toolcall"]