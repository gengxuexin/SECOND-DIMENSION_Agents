"""Agent 公共基类：统一"规划-工具执行-结构化落盘-低置信转人工"的执行骨架。

三个 Agent（图像归档 / 评论洞察 / 运营建议）都复用该骨架，差异只在
system prompt、关注的工具子集与输出 schema，这正是 V2 拆分多 Agent 的边界依据。
"""
from __future__ import annotations

import time
from typing import Any

from schemas import TaskStatus, ToolCallRecord, WorkflowState
from tools.registry import TOOL_SPECS, invoke

BASE_SYSTEM = """你是服装门店与网店智能运营助手系统的编排层（LLM），负责理解意图、选择工具、生成参数与措辞。

硬性规则（违反视为错误输出）：
1. 任何关于"这件衣服是什么品类/款式"的结论，必须来自 classify_image 工具，禁止自己判断。
2. 任何关于"评论是好评/差评、问题出在尺码还是颜色质量"的结论，必须来自 analyze_comments 工具，禁止自己推断。
3. 你只能使用提供的工具，不得假设存在其他工具，不得凭空生成商品、评论、库存数据。
4. 缺少数据时，明确说"需要先执行 XX 工具获取"，不要编造数字。
5. 运营建议必须引用工具输出或具体评论证据，无证据的泛泛而谈不允许输出。
"""


class BaseAgent:
    name = "base"
    system = BASE_SYSTEM
    tool_names: list[str] = []

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.call_records: list[ToolCallRecord] = []

    # ---------------------------------------------------------- 工具子集
    @property
    def tools(self) -> list[dict[str, Any]]:
        if not self.tool_names:
            return TOOL_SPECS
        return [t for t in TOOL_SPECS if t["function"]["name"] in self.tool_names]

    # ---------------------------------------------------------- 主循环
    def run(self, state: WorkflowState, context: dict[str, Any],
            max_rounds: int = 4) -> dict[str, Any]:
        """执行 Agent。返回结构化产出 dict，同时把过程写入 state.trace。"""
        state.status = TaskStatus.RUNNING
        state.log("start", "system", f"agent={self.name} llm={self.llm.name}")

        messages = [
            {"role": "system", "content": _json(context)},
            {"role": "user", "content": state.user_query},
        ]
        observations: list[dict[str, Any]] = []
        rounds = 0
        t0 = time.perf_counter()

        # 每轮把已取回的数据回灌进上下文：编排层据此决定下一步，而不是靠解析历史消息。
        # 这也让真实 LLM 与离线降级实现行为一致（两者都看结构化上下文）。
        live: dict[str, Any] = {}

        messages: list[dict[str, Any]] = []
        while rounds < max_rounds:
            rounds += 1
            # 每轮重建消息：system 槽位换成"任务上下文 + 已取回数据"的最新快照，
            # 保留之前的 assistant/tool 消息让真实 LLM 保持连贯。
            round_ctx = {**context, **live}
            head = [{"role": "system", "content": _json(round_ctx)},
                    {"role": "user", "content": state.user_query}]
            messages = head + messages[len(head):] if len(messages) > len(head) \
                else head + messages[2:]
            try:
                resp = self.llm.chat_with_tools(messages, self.tools, self.system, max_rounds)
            except Exception as e:
                state.errors.append(f"{self.name}: LLM 调用失败 {type(e).__name__}: {e}")
                state.log("llm_error", "llm", str(e))
                break

            calls = resp.get("tool_calls") or []
            if not calls:
                state.log("llm_final", "llm", (resp.get("content") or "")[:200])
                break

            messages.append({"role": "assistant", "content": resp.get("content") or "",
                             "tool_calls": _as_openai_calls(calls)})
            for c in calls:
                out, rec = invoke(c["name"], c.get("arguments") or {})
                self.call_records.append(rec)
                # 记录批量打分入参顺序，供 postprocess 做结果-评论严格对齐
                if c["name"] == "analyze_comments":
                    self._analyze_order = list((c.get("arguments") or {}).get("comments") or [])
                state.log("tool_call", "tool",
                          f"{c['name']} -> {rec.result_summary}",
                          {"args": rec.arguments, "ok": rec.ok}, cost_ms=rec.cost_ms)
                if not rec.ok:
                    state.errors.append(f"{c['name']}: {rec.result_summary}")
                observations.append(out)
                self._absorb(live, c["name"], out)
                messages.append({"role": "tool", "tool_call_id": c["call_id"],
                                 "name": c["name"],
                                 "content": _trim(_json(out))})

        cost = round((time.perf_counter() - t0) * 1000, 2)
        result = self.postprocess(state, observations, context)
        state.log("end", "system", f"agent={self.name} rounds={rounds}", {"cost_ms": cost})
        result["_meta"] = {"agent": self.name, "llm": self.llm.name, "rounds": rounds,
                           "cost_ms": cost, "tool_calls": len(self.call_records)}
        return result

    # ---------------------------------------------------------- 子类实现
    def postprocess(self, state: WorkflowState, observations: list[dict],
                    context: dict) -> dict[str, Any]:
        raise NotImplementedError

    # ---------------------------------------------------------- 上下文累积
    def _absorb(self, live: dict[str, Any], tool_name: str, out: dict) -> None:
        absorb_tool_result(live, tool_name, out)


def absorb_tool_result(live: dict[str, Any], tool_name: str, out: dict) -> None:
    """把工具返回的关键数据回灌到下一轮的上下文。

    这是多轮工具调用的关键：query_comments 拿到评论后，analyze_comments 才拿得到正文。
    无论 LLM 是真实大模型还是离线降级实现，都依赖这份结构化快照，
    因此两者行为一致，回归测试的结论也可迁移。
    """
    if not out.get("ok", True):
        return
    if tool_name == "query_comments":
        live.setdefault("comments", [])
        seen = {c.get("comment_id") for c in live["comments"]}
        for c in out.get("comments", []) or []:
            if c.get("comment_id") not in seen:
                live["comments"].append(c)
    elif tool_name == "query_products":
        live.setdefault("products", [])
        seen = {p.get("sku_id") for p in live["products"]}
        for p in out.get("products", []) or []:
            if p.get("sku_id") not in seen:
                live["products"].append(p)
    elif tool_name == "classify_image":
        live.setdefault("image_predictions", [])
        live["image_predictions"].extend(out.get("predictions", []) or [])
        if out.get("processed_path"):
            live.setdefault("classified_paths", [])
            if out["processed_path"] not in live["classified_paths"]:
                live["classified_paths"].append(out["processed_path"])
    elif tool_name == "normalize_tags":
        live.setdefault("normalized_tags", [])
        live["normalized_tags"].extend(out.get("normalized", []) or [])
    live.setdefault("called_tools", [])
    if tool_name not in live["called_tools"]:
        live["called_tools"].append(tool_name)


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)


def _trim(s: str, limit: int = 1500) -> str:
    """工具返回体裁剪。

    注意：结构化 JSON 被截断后无法解析，会让后续轮次的编排器拿不到已取回的数据。
    因此这里只对"超长且解析不回来"的情况截断，并显式标注 truncated 标记，
    让调用方能识别"数据不完整"，而不是把坏 JSON 当成空结果静默处理。
    """
    if len(s) <= limit:
        return s
    # 尝试在预算内保留尽量多的完整条目（按行/数组元素回退）
    for keep in range(8, 1, -1):
        candidate = _shrink(s, keep)
        if candidate is not None and len(candidate) <= limit:
            return candidate + f"...[truncated_to_{keep}_items]"
    return s[:limit] + f"...<truncated {len(s) - limit} chars, INCOMPLETE>"


def _shrink(s: str, keep: int) -> str | None:
    """尝试把 JSON 裁剪为只保留前 keep 条主要数组元素，失败返回 None。"""
    import json as _j
    try:
        obj = _j.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out = dict(obj)
    changed = False
    for key in ("comments", "results", "products", "predictions"):
        arr = out.get(key)
        if isinstance(arr, list) and len(arr) > keep:
            out[key] = arr[:keep]
            out["_original_count"] = len(arr)
            changed = True
    if not changed:
        return None
    try:
        return _j.dumps(out, ensure_ascii=False, default=str)
    except Exception:
        return None


def _as_openai_calls(calls: list[dict]) -> list[dict]:
    return [{"id": c["call_id"], "type": "function",
             "function": {"name": c["name"], "arguments": _json(c.get("arguments") or {})}}
            for c in calls]