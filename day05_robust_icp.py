import numpy as np
from scipy.spatial import cKDTree

from day03_icp import best_fit_transform, make_points, rotation_z, icp

def trimmed_icp(source,target,max_iterations=60,
        keep_ratio=0.8):
    current=source.copy()
    tree=cKDTree(target)

    total_R=np.eye(3)
    total_t=np.zeros(3)
    errors=[]

    for _ in range(max_iterations):
        distances, indices = tree.query(current)

        order = np.argsort(distances)

# 确定保留多少对匹配
        keep_count = max(3, int(len(current) * keep_ratio))

# 取距离较近的那部分匹配
        keep = order[:keep_count]

# 只用保留的匹配估计变换
        R, t = best_fit_transform(
            current[keep],
            target[indices[keep]]
)

        current=current @ R.T+t

        total_R=R@total_R
        total_t=total_t @ R.T +t

        # 记录本轮更新后的最近点均方根误差
        new_distances, _ = tree.query(current)
        error = np.sqrt(np.mean(new_distances ** 2))
        errors.append(error)

        if len(errors)>1:
            if abs(errors[-2]-errors[-1])<1e-8:
                break

    return current,total_R,total_t,errors

# 计算配准误差，占物体尺度的比例
def relative_rmse(recovered, target, scale):
    rmse = np.sqrt(
        np.mean(np.sum((recovered - target) ** 2, axis=1))
    )
    return rmse / scale


def main():
    # 第一步：建立干净数据，保留作为评价依据
    seed = 0
    target = make_points(seed)

    scale = np.linalg.norm(np.ptp(target, axis=0))

    source_clean = (
        target @ rotation_z(10).T
        + np.array([0.08, -0.04, 0.03])
    )

    # 第二步：复制一份，再把其中 10% 的点替换为乱点
    # 不会改变上面的 source_clean
    outlier_ratio = 0.1
    rng = np.random.default_rng(seed + 2000)

    source_bad = source_clean.copy()
    count = int(len(source_bad) * outlier_ratio)

    bad_indices = rng.choice(
        len(source_bad),
        size=count,
        replace=False
    )

    source_bad[bad_indices] = (
        source_clean.mean(axis=0)
        + rng.uniform(-scale, scale, size=(count, 3))
    )

    # 第三步：普通 ICP 处理含乱点的数据
    _, R_basic, t_basic, errors_basic = icp(
        source_bad, target
    )

    # 第四步：筛选 ICP 处理同一份含乱点的数据
    _, R_trim, t_trim, errors_trim = trimmed_icp(
        source_bad, target,
        keep_ratio=0.8
    )

    # 第五步：将两种算法估计的变换作用于干净数据
    # 这一步是评价，不是重新运行 ICP
    recovered_basic = source_clean @ R_basic.T + t_basic
    recovered_trim = source_clean @ R_trim.T + t_trim

    # 第六步：用相同标准计算误差
    error_basic = relative_rmse(
        recovered_basic, target, scale
    )
    error_trim = relative_rmse(
        recovered_trim, target, scale
    )

    print(f"离群点比例：{outlier_ratio:.0%}")
    print(f"普通 ICP 相对误差：{error_basic:.4%}")
    print(f"筛选 ICP 相对误差：{error_trim:.4%}")
    print("普通 ICP 迭代次数：", len(errors_basic))
    print("筛选 ICP 迭代次数：", len(errors_trim))


# 文件运行时，从 main() 开始执行实验
if __name__ == "__main__":
    main()