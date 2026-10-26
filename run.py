"""命令行入口：一条命令跑通"图片归档 → 标签归一 → 评论洞察 → 运营建议"全链路。

用法：
    python run.py init                      # 初始化演示数据
    python run.py serve                     # 启动两个确定性模型服务（前台）
    python run.py chat "帮我看看这批新衣服" # 自然语言跑多 Agent 闭环
    python run.py classify <图片路径...>     # V1：单 Agent 工具化，直接调分类
    python run.py comments                   # V2：只跑评论洞察
    python run.py advice                     # V2：评论 + 商品 -> 运营建议
    python run.py eval                       # V3：评测回归
    python run.py review                     # V3：查看人工确认队列
    python run.py trace <task_id>            # 查看全链路轨迹
    python run.py train image|comment        # 训练/校准两个确定性模型
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config


# ---------------------------------------------------------------- 辅助
def _ensure_data() -> None:
    if not config.PRODUCT_TABLE.exists():
        import data_seed
        data_seed.write_all()


def _demo_images() -> list[str]:
    """无真实图片时生成样例图，保证链路可端到端演示。

    轮廓按品类绘制（连衣裙 A 字摆、阔腿裤两裤腿、外套带肩线、鞋类扁长），
    这样兜底分类器能从几何特征得到语义合理的结果，而非全靠随机。
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return []
    d = config.BASE_DIR / "data" / "demo_images"
    d.mkdir(parents=True, exist_ok=True)
    W, H = 320, 440

    def bg_fill(bg):
        img = Image.new("RGB", (W, H), bg)
        return img, ImageDraw.Draw(img)

    specs = {
        # name: (bg, fg, 画法)
        "dress_001": ((238, 232, 220), (196, 160, 140), "dress"),
        "coat_002": ((212, 192, 158), (86, 70, 56), "coat"),
        "knit_003": ((168, 168, 172), (66, 66, 70), "knit"),
        "pants_004": ((172, 196, 226), (48, 66, 96), "pants"),
    }

    def draw_dress(dr, fg):
        dr.polygon([(150, 120), (170, 120), (200, 190), (222, 400), (98, 400), (120, 190)], fill=fg)
        dr.ellipse([138, 92, 182, 134], fill=fg)

    def draw_coat(dr, fg):
        dr.rectangle([112, 130, 208, 400], fill=fg)
        dr.polygon([(112, 130), (86, 158), (96, 196), (112, 178)], fill=fg)
        dr.polygon([(208, 130), (234, 158), (224, 196), (208, 178)], fill=fg)
        dr.ellipse([138, 96, 182, 140], fill=fg)

    def draw_knit(dr, fg):
        dr.rounded_rectangle([110, 132, 210, 340], radius=18, fill=fg)
        dr.rectangle([100, 300, 220, 322], fill=fg)
        dr.ellipse([138, 100, 182, 142], fill=fg)

    def draw_pants(dr, fg):
        dr.rectangle([128, 120, 192, 250], fill=fg)
        dr.polygon([(128, 240), (158, 240), (152, 410), (122, 410)], fill=fg)
        dr.polygon([(162, 240), (192, 240), (198, 410), (168, 410)], fill=fg)

    painters = {"dress": draw_dress, "coat": draw_coat,
                "knit": draw_knit, "pants": draw_pants}

    out: list[str] = []
    for name, (bg, fg, kind) in specs.items():
        p = d / f"{name}.png"
        img, dr = bg_fill(bg)
        painters[kind](dr, fg)
        if name.startswith("coat"):
            img = img.rotate(-4, expand=False, fillcolor=bg)
        img.save(p)
        out.append(str(p))
    return out


def _tool_alive(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _check_services() -> bool:
    a, b = _tool_alive(config.IMAGE_SERVICE_URL), _tool_alive(config.COMMENT_SERVICE_URL)
    print(f"[health] 图像分类服务 {config.IMAGE_SERVICE_URL}: {'up' if a else 'DOWN'}")
    print(f"[health] 评论分析服务 {config.COMMENT_SERVICE_URL}: {'up' if b else 'DOWN'}")
    if not (a and b):
        print("[health] 提示：先执行 `python run.py serve` 启动工具层服务")
    return a and b


# ---------------------------------------------------------------- 子命令
def cmd_init(_: argparse.Namespace) -> int:
    import data_seed
    data_seed.write_all()
    imgs = _demo_images()
    if imgs:
        print(f"样例图片已就绪：{len(imgs)} 张 -> data/demo_images/")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import threading

    import services.comment_service as csvc
    import services.image_service as imsvc
    threading.Thread(target=imsvc.serve, args=(args.image_port,), daemon=True).start()
    csvc.serve(args.comment_port)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    _ensure_data()
    _check_services()
    from orchestration import MultiAgentWorkflow
    paths = _demo_images() if args.images else []
    scope = "all"
    for tok in (args.images or "").split(","):
        if "品类:" in tok:
            scope = tok.split("品类:")[1].strip() or "all"
    wf = MultiAgentWorkflow()
    st = wf.run(args.query, image_paths=paths, comment_scope=scope)
    print(st.report)
    _print_governance(st)
    print(f"\n[trace] task_id={st.task_id}　节点轨迹 "
          f"{' -> '.join(e.step for e in st.trace)}")
    (config.VAR_DIR / f"trace_{st.task_id}.json").write_text(
        st.model_dump_json(indent=2), encoding="utf-8")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    """V1：Function Calling 工具化路径。"""
    _ensure_data()
    _check_services()
    paths = [str(Path(p).resolve()) for p in args.paths]
    if not paths:
        paths = _demo_images()
    from orchestration import single_agent_toolcall
    res = single_agent_toolcall(
        "把这几张新上的衣服图片归类，返回标准标签",
        {"image_paths": paths})
    print(f"[v1] 编排后端={res['llm']}  工具调用 {len(res['calls'])} 次")
    for c in res["calls"]:
        print(f"  · {c['name']}({json.dumps(c['args'], ensure_ascii=False)[:70]}) "
              f"-> {c['summary']}  [{c['cost_ms']}ms, ok={c['ok']}]")
        if c["name"] == "classify_image":
            for p in c["result"].get("predictions", []):
                print(f"      #{p['top_k_rank']} {p['label']}  score={p['score']:.4f}")
            err = c["result"].get("error")
            if err:
                print(f"      ! {err['code']}: {err['message']}")
    return 0


def cmd_comments(args: argparse.Namespace) -> int:
    _ensure_data()
    _check_services()
    from orchestration import MultiAgentWorkflow
    st = MultiAgentWorkflow().run(
        "看看最近评论里尺码、颜色、质量方面有什么问题", comment_scope=args.scope)
    print(st.report)
    _print_governance(st)
    return 0


def cmd_advice(args: argparse.Namespace) -> int:
    _ensure_data()
    _check_services()
    from orchestration import MultiAgentWorkflow
    st = MultiAgentWorkflow().run("根据评论和库存给我一些选品和售后建议",
                                 image_paths=[], comment_scope=args.scope)
    print(st.report)
    _print_governance(st)
    (config.VAR_DIR / f"trace_{st.task_id}.json").write_text(
        st.model_dump_json(indent=2), encoding="utf-8")
    return 0


def cmd_eval(_: argparse.Namespace) -> int:
    from eval.regression import print_report, run_eval
    _check_services()
    res = run_eval()
    print_report(res)
    (config.VAR_DIR / "eval_report.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] var/eval_report.json")
    return 0 if res["overall"] >= 0.8 else 1


def cmd_review(_: argparse.Namespace) -> int:
    import store
    rows = store.load_human_samples()
    if not rows:
        print("人工回流样本池为空。运行一次带低置信项的任务后可用 `AUTO_CONFIRM=1` 灌入样本。")
        return 0
    print(f"人工修正回流样本：{len(rows)} 条")
    for r in rows[-10:]:
        print(f"  [{r.get('kind')}] {r.get('reason')} -> {r.get('decision')} "
              f"(corrected={r.get('corrected_value')}) by={r.get('by')}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    p = config.VAR_DIR / f"trace_{args.task_id}.json"
    if not p.exists():
        cands = sorted(config.VAR_DIR.glob("trace_*.json"))
        if not cands:
            print("暂无轨迹文件，先运行 chat/advice 任务")
            return 1
        p = cands[-1]
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"task={d['task_id']} intent={d['intent']} status={d['status']}")
    for e in d["trace"]:
        print(f"  [{e['step']:<12}] {e['actor']:<6} {e['detail']} ({e['cost_ms']}ms)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """把 train 子命令的剩余参数原样透传给训练器。

    训练参数（--data/--epochs/--lr/...）由训练器自己定义，
    这里不重复声明，避免两处参数定义不同步。
    """
    import sys as _s

    argv = _s.argv[1:]
    # 去掉 "train" 本身，保留 "image|comment|fmn" 与其后的所有参数
    if "train" in argv:
        argv = argv[argv.index("train") + 1:]

    if argv and argv[0] == "fmn":
        import services.fashion_mnist_model as fmn
        ap = argparse.ArgumentParser(description="Fashion-MNIST 小型 CNN 训练")
        ap.add_argument("--epochs", type=int, default=5)
        ap.add_argument("--train-limit", type=int, default=12000)
        ap.add_argument("--test-limit", type=int, default=3000)
        ap.add_argument("--batch-size", type=int, default=64)
        ap.add_argument("--lr", type=float, default=1e-3)
        a = ap.parse_args(argv[1:])
        fmn.train(epochs=a.epochs, train_limit=a.train_limit, test_limit=a.test_limit,
                  lr=a.lr, batch_size=a.batch_size)
        return 0

    import services.train_models as tm
    _s.argv = ["train_models.py"] + argv
    tm.main()
    return 0


def cmd_eval_model(args: argparse.Namespace) -> int:
    """Fashion-MNIST 模型评测：在官方测试集上跑混淆矩阵 + 每类指标。

    这是 V3"评测与可信治理"里模型层的评测——Agent 编排层评测之外，
    确定性模型本身也要有可复现的量化指标，而不是嘴上说"准确率 92%"。
    """
    from services.fashion_mnist_model import CLASSES, CATEGORY_CN, load_split
    import services.fashion_mnist_model as fmn

    if not fmn.WEIGHT_PATH.exists():
        print(f"[eval-model] 权重不存在：{fmn.WEIGHT_PATH}")
        print("[eval-model] 先训练：python run.py train fmn --epochs 6")
        return 1

    xte, yte = load_split("test", limit=args.limit)
    model = fmn._load_torchscript(fmn.WEIGHT_PATH)
    import torch
    from PIL import Image
    import numpy as np

    preds = []
    for i in range(len(xte)):
        arr = np.asarray(xte[i], dtype="float32").reshape(28, 28)
        t = torch.from_numpy(arr).view(1, 1, 28, 28)
        with torch.no_grad():
            preds.append(int(model(t).argmax(1)))

    # 混淆矩阵
    n_cls = len(CLASSES)
    cm = [[0] * n_cls for _ in range(n_cls)]
    for p, y in zip(preds, yte):
        cm[y][p] += 1
    total = len(yte)
    correct = sum(cm[i][i] for i in range(n_cls))
    acc = correct / max(total, 1)

    print("=" * 68)
    print(f"Fashion-MNIST 模型评测　测试集 {total} 张　总体准确率 {acc:.2%}")
    print("=" * 68)
    print(f"{'类别':<16}{'中文标签':<8}{'准确率':>9}{'召回率':>9}{'F1':>9}{'样本':>7}")
    print("-" * 68)
    for i, cls in enumerate(CLASSES):
        tp = cm[i][i]
        support = sum(cm[i])
        pred_n = sum(cm[r][i] for r in range(n_cls))
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"{cls:<16}{CATEGORY_CN[cls]:<8}{prec:>9.2%}{rec:>9.2%}{f1:>9.2%}{support:>7}")
    print("-" * 68)
    print(f"{'总体':<16}{'':<8}{'':>9}{'':>9}{'':>9}{total:>7}　准确率 {acc:.2%}")
    print("=" * 68)
    print("提示：该模型基于 28x28 灰度图，对真实商品照片存在域差")
    print("     （实测实拍图 0/4），实拍场景请用 prototype-nn 或 siglip2 后端。")
    return 0


def cmd_validate_backends(args: argparse.Namespace) -> int:
    """后端可信度验证：用业务数据实测每个可用后端，生成 var/backend_trust.json。

    为什么要这一步：模型的"公开指标"不等于"在你的场景能用"。
    实测 SigLIP2 在 Fashion-MNIST 官方测试集 91.8%，但对真实商品照片 0/4，
    且以 0.98 的置信度给出完全错误的结论——比低置信错误更危险。

    验证方式：用 data/train_images/ 下按标准标签分目录的样本做留出评测，
    只有实测准确率 >= 阈值的后端才会被采信（参与交叉验证投票）。
    """
    import json

    import services.image_service as svc

    root = config.BASE_DIR / args.data
    if not root.exists():
        print(f"[validate] 数据目录不存在：{root}")
        print(f"[validate] 目录约定：每个子目录名 = 标准标签，目录内是图片")
        return 1

    # 收集带标签样本
    samples: list[tuple[str, str]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                samples.append((str(f), sub.name))
    if not samples:
        print(f"[validate] {root} 下没有找到图片")
        return 1

    # 后端清单：哪个可用就测哪个
    backends = []
    if (svc.SIGLIP_DIR / "model.safetensors").exists() or (svc.SIGLIP_DIR / "pytorch_model.bin").exists():
        backends.append("siglip2")
    if svc.FASHION_CNN_PATH.exists():
        backends.append("fashion-cnn")
    if svc.PROTOS:
        backends.append("prototype-nn")
    if not backends:
        print("[validate] 没有可用后端（缺少权重/原型库）")
        return 1

    print(f"[validate] 样本 {len(samples)} 张，标签 {len(set(t for _, t in samples))} 类")
    print(f"[validate] 待测后端：{', '.join(backends)}\n")

    results: dict[str, dict] = {}
    for backend in backends:
        # 逐样本强制用指定后端推理
        saved = svc.BACKEND
        svc.BACKEND = backend
        correct = 0
        total = 0
        errors = 0
        details = []
        for path, truth in samples:
            try:
                res = svc.predict(path, 3)
            except Exception as e:
                errors += 1
                details.append(f"{Path(path).name}: 推理异常 {type(e).__name__}")
                continue
            if not res.get("ok") or not res.get("predictions"):
                errors += 1
                details.append(f"{Path(path).name}: 无预测结果")
                continue
            total += 1
            top = res["predictions"][0]["label"]
            hit = top == truth
            correct += int(hit)
            if not hit:
                details.append(f"{Path(path).name}({truth}): 判为 {top}")
        svc.BACKEND = saved
        acc = correct / max(total, 1)
        trusted = acc >= args.min_acc
        results[backend] = {
            "trusted": trusted, "measured_acc": round(acc, 4),
            "correct": correct, "total": total, "errors": errors,
            "min_acc_required": args.min_acc,
            "detail": details[:10],
        }
        flag = "✓ 采信" if trusted else "✗ 不采信"
        print(f"[validate] {backend:<14} 准确率 {acc:>6.1%} ({correct}/{total})  {flag}")
        for d in details[:5]:
            print(f"             - {d}")
        print()

    out = config.VAR_DIR / "backend_trust.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[validate] 可信度注册表已写入 -> {out}")
    trusted_list = [k for k, v in results.items() if v["trusted"]]
    print(f"[validate] 采信后端：{trusted_list or '（无，全部不采信）'}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """启动 Web 前端 + API（AdminLTE 4 开源前端，见 web/vendor/）。"""
    _ensure_data()
    from web.api import serve
    print(f"[web] 前端资源: web/index.html  后端接口: /api/*")
    print(f"[web] 请确保模型服务已启动（另开终端 python run.py serve）")
    _check_services()
    serve(args.port, args.host)
    return 0


def _print_governance(st) -> None:
    from governance.trust import ReviewQueue, triage, validate_evidence
    tg = triage(st)
    vr = validate_evidence(st)
    q = ReviewQueue()
    items = q.from_state(st)
    print("\n" + "-" * 68)
    print(f"[治理] 自动采信率 {tg['auto_rate']:.1%}　"
          f"待人工确认 {len(tg['need_human'])} 项　"
          f"证据闭环 {'通过' if vr['hallucination_free'] else '未通过'}")
    if items:
        print(f"[治理] 人工确认队列 {len(items)} 条（等待店主/运营处理）")
        for it in items[:5]:
            print(f"   - {it['item_id']} {it['kind']}: {it['reason']}")


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run.py", description="服装门店与网店智能运营多Agent助手")
    ap.add_argument("--query", default="帮我把这批新上的衣服归类，并看看网店评论有什么问题，最后给点运营建议")
    ap.add_argument("--images", default="", help="图片路径，逗号分隔；支持 品类:连衣裙 指定评论范围")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("init", help="初始化演示数据").set_defaults(func=cmd_init)
    p = sub.add_parser("serve", help="启动两个确定性模型服务")
    p.add_argument("--image-port", type=int, default=8801)
    p.add_argument("--comment-port", type=int, default=8802)
    p.set_defaults(func=cmd_serve)

    sub.add_parser("chat", help="自然语言跑多Agent闭环").set_defaults(func=cmd_chat)
    p = sub.add_parser("classify", help="V1 单Agent工具化")
    p.add_argument("paths", nargs="*", help="图片路径；留空则使用样例图")
    p.set_defaults(func=cmd_classify)
    p = sub.add_parser("comments", help="V2 评论洞察")
    p.add_argument("--scope", default="all")
    p.set_defaults(func=cmd_comments)
    p = sub.add_parser("advice", help="V2 运营建议")
    p.add_argument("--scope", default="all")
    p.set_defaults(func=cmd_advice)
    sub.add_parser("eval", help="V3 评测回归").set_defaults(func=cmd_eval)
    sub.add_parser("review", help="V3 人工回流样本").set_defaults(func=cmd_review)
    p = sub.add_parser("trace", help="查看链路轨迹")
    p.add_argument("task_id", nargs="?")
    p.set_defaults(func=cmd_trace)
    p = sub.add_parser("train", help="训练/校准模型（其余参数透传给训练器）")
    p.add_argument("what", nargs="?", choices=["image", "comment"])
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("web", help="启动 Web 前端（AdminLTE 4）")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(func=cmd_web)
    p = sub.add_parser("eval-model", help="评测 Fashion-MNIST 模型（混淆矩阵 + 每类指标）")
    p.add_argument("--limit", type=int, default=3000)
    p.set_defaults(func=cmd_eval_model)
    p = sub.add_parser("validate-backends",
                       help="用业务数据验证各后端，生成可信度注册表")
    p.add_argument("--data", default="data/train_images")
    p.add_argument("--min-acc", type=float, default=0.60)
    p.set_defaults(func=cmd_validate_backends)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    ap = build_parser()
    if not argv:
        return int(cmd_chat(ap.parse_args([])) or 0)
    # 无子命令时（只给了全局参数）默认走 chat
    known = {"init", "serve", "chat", "classify", "comments",
             "advice", "eval", "review", "trace", "train", "web", "eval-model",
             "validate-backends"}
    if not any(a in known for a in argv):
        argv.append("chat")
    if "train" in argv:
        # train 的训练参数由训练器定义，用 parse_known_args 放行
        args, _unknown = ap.parse_known_args(argv)
    else:
        args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())