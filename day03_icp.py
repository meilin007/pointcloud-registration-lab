import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def best_fit_transform(A,B):
    center_A=A.mean(axis=0)
    center_B=B.mean(axis=0)

    AA=A-center_A
    BB=B-center_B

    H=AA.T @ BB
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T @ U.T

    if np.linalg.det(R)<0:
        Vt[-1,:]*=-1
        R=Vt.T @ U.T

    t=center_B-center_A @ R.T
    return R,t

def icp(source,target,max_iterations=60):
    current=source.copy()
    tree=cKDTree(target)

    total_R=np.eye(3)
    total_t=np.zeros(3)
    errors=[]

    for _ in range(max_iterations):
        distance,indices=tree.query(current)

        R,t=best_fit_transform(current,target[indices])

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


def make_points(seed=42):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, size=(300, 3))
    points *= np.array([1.0, 0.6, 0.3])
    return points


# 生成绕 z 轴旋转的矩阵
def rotation_z(degrees):
    angle = np.deg2rad(degrees)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])


def draw_pair(ax, target, moving, title):
    ax.scatter(*target.T, s=8, color="blue", label="Target")
    ax.scatter(*moving.T, s=8, color="orange", label="Moving")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 两张图使用相同范围，方便比较
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect((1, 1, 1))
    ax.legend()


def main():
    target = make_points()

    # 人为制造已知的旋转和平移
    angle=60
    R_true=rotation_z(angle)
    t_true = np.array([0.08, -0.04, 0.03])
    source = target @ R_true.T + t_true

    aligned, R_est, t_est, errors = icp(source, target)

    # 这里保留原始点顺序，便于检查真实对应点误差。
    # ICP 本身没有使用这些已知对应关系。
    paired_rmse = np.sqrt(
        np.mean(np.sum((aligned - target) ** 2, axis=1))
    )

    print("迭代次数：", len(errors))
    print("最终最近点 RMSE", errors[-1])
    print("真实对应点 RMSE", paired_rmse)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    draw_pair(ax1, target, source, "Before ICP")
    draw_pair(ax2, target, aligned, "After ICP")

    plt.tight_layout()
    plt.savefig("day03_alignment_{angle}deg.png", dpi=180)

    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Nearest-point RMSE")
    plt.title("ICP convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("day03_error_{angle}deg.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
