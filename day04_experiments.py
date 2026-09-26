from day03_icp import make_points,rotation_z,icp

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / "day04_results"

def run_one(angle,noise_ratio,seed):
    target=make_points(seed)

    scale=np.linalg.norm(np.ptp(target,axis=0))

    R_true=rotation_z(angle)
    t_true=np.array([0.08,-0.04,0.03])

    source_clean=target @ R_true.T +t_true

    rng=np.random.default_rng(seed+1000)
    noise=rng.normal(
        0,
        noise_ratio*scale,
        size=source_clean.shape
    )
    source_observed=source_clean+noise

    __,R_est,t_est,errors=icp(source_observed,target)

    recovered=source_clean @ R_est.T + t_est

    rmse=np.sqrt(
        np.mean(np.sum((recovered-target)**2,axis=1))
    )
    relative_error=rmse/scale

    return{
        "angle_deg": angle,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "relative_error": float(relative_error),
        "iterations": len(errors),
        "success": int(relative_error < 0.01)
    }

def main():
    OUTPUT.mkdir(exist_ok=True)
    rows = []

    # 实验 A：改变初始角度
    for angle in [5, 15, 30, 60]:
        for seed in range(10):
            result = run_one(angle, 0.0, seed)
            result["experiment"] = "angle"
            rows.append(result)

        print(f"角度 {angle} 度：完成 10 次")

    # 实验 B：改变测量噪声
    for noise_ratio in [0.0, 0.005, 0.01, 0.02]:
        for seed in range(10):
            result = run_one(10, noise_ratio, seed)
            result["experiment"] = "noise"
            rows.append(result)

        print(f"噪声 {noise_ratio:.1%}：完成 10 次")

    # 保存每次实验的原始结果
    fields = [
        "experiment", "angle_deg", "noise_ratio",
        "seed", "relative_error", "iterations", "success"
    ]

    with open(
        OUTPUT / "results.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 图 1：不同初始角度下的成功率
    angles = [5, 15, 30, 60]
    rates = []

    for angle in angles:
        selected = [
            row for row in rows
            if row["experiment"] == "angle"
            and row["angle_deg"] == angle
        ]
        rates.append(
            100 * np.mean([row["success"] for row in selected])
        )

    plt.figure()
    plt.plot(angles, rates, marker="o")
    plt.xticks(angles)
    plt.ylim(-5, 105)
    plt.xlabel("Initial rotation (degrees)")
    plt.ylabel("Success rate (%)")
    plt.title("ICP: effect of initial rotation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT / "angle_success.png", dpi=180)
    plt.close()

    # 图 2：不同噪声下的误差
    noise_levels = [0.0, 0.005, 0.01, 0.02]
    means = []
    stds = []

    for level in noise_levels:
        values = [
            row["relative_error"] * 100
            for row in rows
            if row["experiment"] == "noise"
            and row["noise_ratio"] == level
        ]
        means.append(np.mean(values))
        stds.append(np.std(values, ddof=1))

    plt.figure()
    plt.errorbar(
        np.array(noise_levels) * 100,
        means,
        yerr=stds,
        marker="o",
        capsize=4
    )
    plt.xlabel("Noise standard deviation / object scale (%)")
    plt.ylabel("Registration error / object scale (%)")
    plt.title("ICP: mean error and standard deviation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT / "noise_error.png", dpi=180)
    plt.close()

    print("全部完成。结果位置：", OUTPUT)


if __name__ == "__main__":
    main()
