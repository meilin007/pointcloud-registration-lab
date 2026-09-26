"""四种离群点比例，每种十个种子；两种方法共用相同输入和评价点。"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from day03_icp import make_points, rotation_z, icp
from day05_robust_icp import trimmed_icp, relative_rmse

OUTPUT = Path(__file__).resolve().parent / "day05_results"


def main():
    OUTPUT.mkdir(exist_ok=True)
    rows = []
    levels = [0.0, 0.05, 0.10, 0.20]
    for ratio in levels:
        for seed in range(10):
            target = make_points(seed)
            scale = np.linalg.norm(np.ptp(target, axis=0))
            clean = target @ rotation_z(10).T + np.array([0.08, -0.04, 0.03])
            observed = clean.copy()
            rng = np.random.default_rng(seed + 2000)
            count = int(len(clean) * ratio)
            indices = rng.choice(len(clean), count, replace=False)
            observed[indices] = clean.mean(axis=0) + rng.uniform(-scale, scale, (count, 3))
            for name, method in [("Basic ICP", icp), ("Trimmed ICP (80%)", trimmed_icp)]:
                _, R, t, errors = method(observed.copy(), target)
                error = relative_rmse(clean @ R.T + t, target, scale)
                assert np.isfinite(error) and len(errors) > 0
                rows.append(dict(outlier_ratio=ratio, seed=seed, method=name,
                                 relative_error=error, iterations=len(errors),
                                 success=int(error < 0.01)))
        print(f"离群点 {ratio:.0%}：两种方法各完成 10 次")

    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, label, color, marker in [
        ("Basic ICP", "普通 ICP", "#0D6080", "o"),
        ("Trimmed ICP (80%)", "匹配筛选 ICP（保留80%）", "#D87829", "s")
    ]:
        means, stds = [], []
        for ratio in levels:
            group = [r for r in rows if r["method"] == name and r["outlier_ratio"] == ratio]
            values = np.array([r["relative_error"] for r in group]) * 100
            mean, std = float(values.mean()), float(values.std(ddof=1))
            means.append(mean)
            stds.append(std)
            summary = dict(outlier_percent=ratio*100, method=name,
                           mean_error_percent=mean, std_error_percent=std,
                           success_count=sum(r["success"] for r in group),
                           mean_iterations=float(np.mean([r["iterations"] for r in group])))
            summaries.append(summary)
            print(f"{label}，离群点 {ratio:.0%}：平均误差 {mean:.6g}%，成功 {summary['success_count']}/10")
        ax.errorbar(np.array(levels)*100, means, yerr=stds, label=label,
                    color=color, marker=marker, capsize=5, linewidth=2)
    ax.set(xlabel="离群点比例（%）", ylabel="配准误差 / 物体尺度（%）",
           title="离群点对配准精度的影响", xticks=[0, 5, 10, 20])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.015, "合成数据；初始旋转10°；每个条件10次；误差棒为±1个标准差", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUTPUT / "outlier_comparison.png", dpi=200)
    plt.close(fig)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    assert len(rows) == 80
    print("结果目录：", OUTPUT)


if __name__ == "__main__":
    main()
