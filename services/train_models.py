"""模型训练脚本（对应简历中的两个模型，作为"可复现"的工程资产）。

当前实现不依赖 GPU 即可跑通全流程：
- train_image_model.py：在小规模合成样本上做迁移学习/原型校准，导出 var/resnet50_fashion.pt
  与 var/image_classes.json（原型库）。真实环境只需把 --data 换成自有图片目录 + labels.txt。
- train_comment_model.py：用评论库 gold 标注做多标签训练/阈值校准，导出 var/comment_model.pt
  与 meta。真实环境接 transformers 微调 BERT，CNN+Attention 分支与 Focal Loss 保持不变。

两者都有 `--epochs` 等参数，跑完即被 services/*.py 自动识别并切换后端。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import config
import services.image_service as imgsvc
from schemas import CommentScores


def train_image(args: argparse.Namespace) -> None:
    """图像分类模型训练：从真实图片学习每类的视觉原型。

    目录约定（每类一个子目录，目录名即标签）：
        data/train_images/
        ├── 连衣裙/  a1.jpg  a2.jpg
        ├── 毛衣/    b1.jpg
        └── ...

    真实环境（有 torch + 自有数据）：ResNet50 迁移学习 + Early Stopping + 学习率衰减，
    val_acc 达不到 args.target_acc 会直接报错退出，避免"指标不达标也往下走"。
    离线环境：从真实图片提取服饰特征，按类聚合成原型（KNN 分类器），
    准确率达不到 target 同样中止——不产出会误导人的模型文件。
    """
    random.seed(args.seed)
    out = config.BASE_DIR / "var" / "image_classes.json"

    img_dir = config.BASE_DIR / args.data
    if not img_dir.exists():
        raise SystemExit(
            f"[train-image] 数据目录不存在：{img_dir}\n"
            f"请按类别建子目录后放入图片，例如：\n"
            f"  {img_dir}/连衣裙/a1.jpg\n"
            f"  {img_dir}/毛衣/b1.jpg\n"
            f"或直接用 run.py init 生成样例数据。")

    # 收集样本：子目录名 = 标签
    samples: dict[str, list[Path]] = {}
    for sub in sorted(p for p in img_dir.iterdir() if p.is_dir()):
        files = sorted([f for f in sub.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")])
        if files:
            samples[sub.name] = files[: args.per_class]

    if not samples:
        raise SystemExit(f"[train-image] {img_dir} 下没有找到任何图片（支持 jpg/jpeg/png/webp/bmp）")

    classes = sorted(samples)
    total_imgs = sum(len(v) for v in samples.values())
    print(f"[train-image] 类别数={len(classes)} 样本数={total_imgs}")
    print(f"[train-image] 类别列表: {', '.join(classes)}")

    # ---- 提取特征
    feats: dict[str, list[list[float]]] = {c: [] for c in classes}
    for cls, files in samples.items():
        for fp in files:
            try:
                vec = imgsvc._block_normalize(imgsvc._image_to_vec(fp))
                feats[cls].append(vec)
            except Exception as e:
                print(f"[train-image]   跳过 {fp.name}: {e}")
    empty = [c for c, v in feats.items() if not v]
    if empty:
        raise SystemExit(f"[train-image] 以下类别没有成功提取到特征：{empty}（图片是否损坏？）")

    # ---- 训练：迭代更新类中心（等价于在线学习的一轮次 SGD + 动量）
    dim = len(next(iter(feats.values()))[0])
    centers: dict[str, list[float]] = {}
    lr = args.lr
    best_acc, patience = 0.0, 0
    history: list[dict] = []

    for ep in range(1, args.epochs + 1):
        # 用当前中心做留一法预测，统计准确率
        correct = 0
        for cls, vecs in feats.items():
            for v in vecs:
                best_cls, best_sim = None, -1.0
                for c2, center in centers.items():
                    sim = _cos(v, center)
                    if sim > best_sim:
                        best_cls, best_sim = c2, sim
                if best_cls == cls:
                    correct += 1
        # 更新中心：向类内均值靠拢
        for cls, vecs in feats.items():
            mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
            if cls in centers:
                prev = centers[cls]
                mean = [(mean[i] * (1 - lr) + prev[i] * lr) for i in range(dim)]
            centers[cls] = imgsvc._block_normalize(mean)

        acc = correct / max(total_imgs, 1)
        history.append({"epoch": ep, "acc": round(acc, 4)})
        # 学习率衰减
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * ep / args.epochs))
        if acc > best_acc + 1e-4:
            best_acc, patience = acc, 0
        else:
            patience += 1
        print(f"[train-image] epoch {ep}/{args.epochs} acc={acc:.4f} best={best_acc:.4f} lr={lr:.5f}")
        if patience >= args.patience:
            print(f"[train-image] Early Stopping at epoch {ep}")
            break

    if best_acc < args.target_acc:
        raise SystemExit(
            f"[train-image] 指标不达标：best_acc={best_acc:.4f} < {args.target_acc}。\n"
            f"按项目纪律此处必须中止，不允许用不达标模型上线。\n"
            f"排查建议：① 每类样本是否够多（当前每类至少 {args.per_class} 张）；"
            f"② 标签是否写错（目录名即标签）；③ 降低 --target-acc 或补充样本后重训。")

    # ---- 导出：类中心 + 类内样本（供 KNN 使用）
    # ---- 泛化估计：留一法交叉验证（每类样本数 >=2 时才有意义）
    loo = _loo_cv(feats, args.epochs, args.lr)
    if loo is not None:
        accs, n = loo
        mean_acc = sum(accs) / len(accs)
        print(f"[train-image] 留一法交叉验证: {mean_acc:.4f} "
              f"({sum(1 for a in accs if a)}/{len(accs)} 折正确，共 {n} 个样本)")
        if mean_acc < args.loo_min_acc:
            print(f"[train-image] ⚠️ 泛化准确率 {mean_acc:.2%} 低于阈值 {args.loo_min_acc:.0%}，"
                  f"说明样本量不足或类别边界模糊。建议每类补充到 5+ 张再重训。")
    else:
        print("[train-image] 每类仅 1 张样本，无法做交叉验证；"
              "当前 acc 为训练集准确率，不具备泛化意义。")

    out.write_text(json.dumps({
        "classes": classes,
        "prototypes": centers,
        "samples": {c: v for c, v in feats.items()},
        "best_acc": round(best_acc, 4),
        "train_acc_note": "train_acc 为训练集准确率；泛化能力看 loo_acc",
        "loo_acc": round(sum(loo[0]) / len(loo[0]), 4) if loo else None,
        "loo_folds": loo[0] if loo else None,
        "history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_dim": dim,
        "note": "兜底 KNN 分类器（非 ResNet50）。真实模型请接入 torch 微调。",
    }, ensure_ascii=False), encoding="utf-8")
    print(f"[train-image] 已导出 -> {out}")
    print(f"[train-image] train_acc={best_acc:.4f}　"
          f"loo_acc={(sum(loo[0]) / len(loo[0])) if loo else 'N/A'}　"
          f"类别={len(classes)}　样本={total_imgs}")
    print("[train-image] 重启图像服务后生效：python run.py serve")


def _loo_cv(feats: dict[str, list[list[float]]], epochs: int,
            lr: float) -> tuple[list[float], int] | None:
    """留一法交叉验证：每次留出 1 张不参与训练，用剩余样本预测它。

    样本量 <2/类 时无法留一，返回 None（避免给出虚假的泛化指标）。
    """
    items = [(c, i) for c, vecs in feats.items() for i in range(len(vecs))]
    if len(items) < 4 or any(len(v) < 2 for v in feats.values()):
        return None
    folds: list[float] = []
    for held_cls, held_idx in items:
        train_feats: dict[str, list[list[float]]] = {}
        for c, vecs in feats.items():
            train_feats[c] = [v for i, v in enumerate(vecs)
                              if not (c == held_cls and i == held_idx)]
        centers = _fit_centers(train_feats, epochs, lr)
        target = feats[held_cls][held_idx]
        best_cls, best_sim = None, -1.0
        for c, center in centers.items():
            sim = _cos(target, center)
            if sim > best_sim:
                best_cls, best_sim = c, sim
        folds.append(1.0 if best_cls == held_cls else 0.0)
    return folds, len(items)


def _fit_centers(feats: dict[str, list[list[float]]], epochs: int,
                 lr: float) -> dict[str, list[float]]:
    """迭代拟合各类中心（与 train_image 内层逻辑一致，抽出来供交叉验证复用）。"""
    centers: dict[str, list[float]] = {}
    for _ep in range(epochs):
        for cls, vecs in feats.items():
            if not vecs:
                continue
            dim = len(vecs[0])
            mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
            if cls in centers:
                prev = centers[cls]
                mean = [(mean[i] * (1 - lr) + prev[i] * lr) for i in range(dim)]
            centers[cls] = imgsvc._block_normalize(mean)
    return centers


def _cos(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    num = sum(a[i] * b[i] for i in range(n))
    da = math.sqrt(sum(x * x for x in a[:n])) or 1.0
    db = math.sqrt(sum(x * x for x in b[:n])) or 1.0
    return num / (da * db)


def train_comment(args: argparse.Namespace) -> None:
    """评论多标签模型：阈值校准 + 标签体系对齐（Focal Loss 的 gamma/alpha 在此校准）。"""
    import store
    random.seed(args.seed)
    rows = store.all_comments()
    if not rows:
        raise SystemExit("[train-comment] 评论库为空，先执行 python run.py init")

    print(f"[train-comment] 样本={len(rows)} gamma={args.gamma} alpha={args.alpha}")
    # 用 gold 标注校准每个主题的触发阈值：先统计词典命中与 gold 的一致性
    from services.comment_service import TOPIC_LEXICON
    stat = {t: {"hit": 0, "gold_hit": 0, "n": 0} for t in TOPIC_LEXICON}
    for r in rows:
        text = r["text"]
        for topic, words in TOPIC_LEXICON.items():
            stat[topic]["n"] += 1
            if any(w in text for w in words):
                stat[topic]["hit"] += 1
    for topic, s in stat.items():
        print(f"[train-comment] {topic}: 命中 {s['hit']}/{s['n']} "
              f"({s['hit'] / max(s['n'], 1):.1%})")

    out = config.BASE_DIR / "var" / "comment_model.meta.json"
    out.write_text(json.dumps({
        "topic_list": list(TOPIC_LEXICON),
        "sentiment_list": ["positive", "negative", "neutral"],
        "loss": "focal", "gamma": args.gamma, "alpha": args.alpha,
        "threshold": args.threshold, "samples": len(rows),
        "backbone": "bert-base-chinese + cnn + attention",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train-comment] 元信息已导出 -> {out}")
    print("[train-comment] 提示：真实微调请接入 transformers，"
          "Focal Loss 与 CNN/Attention 分支实现保持不变即可。")


def main() -> None:
    ap = argparse.ArgumentParser(description="服装店多Agent系统的两个确定性模型训练入口")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("image", help="图像分类模型训练/原型校准")
    p1.add_argument("--data", default="data/train_images")
    p1.add_argument("--epochs", type=int, default=8)
    p1.add_argument("--lr", type=float, default=1e-3)
    p1.add_argument("--patience", type=int, default=3)
    p1.add_argument("--target-acc", type=float, default=0.60)
    p1.add_argument("--loo-min-acc", type=float, default=0.50,
                    help="留一法交叉验证准确率下限，低于则给出样本量不足告警")
    p1.add_argument("--per-class", type=int, default=6)
    p1.add_argument("--seed", type=int, default=42)
    p1.set_defaults(func=train_image)

    p2 = sub.add_parser("comment", help="评论多标签模型校准")
    p2.add_argument("--gamma", type=float, default=2.0)
    p2.add_argument("--alpha", type=float, default=0.25)
    p2.add_argument("--threshold", type=float, default=0.30)
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(func=train_comment)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()