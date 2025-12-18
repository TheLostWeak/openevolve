import numpy as np
import sys
sys.path.insert(0, "examples/kissing_number_11")
from sota import sphere_centers

print("=== SOTA 解分析（R^11 Kissing Number）===\n")

# 基本信息
n_points = len(sphere_centers)
dim = sphere_centers.shape[1]
print(f"点数: {n_points}")
print(f"维度: {dim}\n")

# 范数统计
norms_squared = np.sum(sphere_centers**2, axis=1)
norms = np.sqrt(norms_squared)
print(f"范数统计:")
print(f"  最小范数: {norms.min():.2f}")
print(f"  最大范数: {norms.max():.2f}")
print(f"  平均范数: {norms.mean():.2f}")
print(f"  标准差: {norms.std():.2f}")
print(f"  范数平方唯一值数: {len(np.unique(norms_squared))}\n")

# 检查范数是否统一
if len(np.unique(norms_squared)) == 1:
    print(" 所有点在同一球面上（统一范数）")
    common_norm_sq = norms_squared[0]
    print(f"  公共范数平方: {common_norm_sq}\n")
else:
    print(" 点不在同一球面上（范数不统一）")
    unique_norms_sq = np.unique(norms_squared)
    print(f"  范数平方的唯一值: {len(unique_norms_sq)}")
    if len(unique_norms_sq) <= 5:
        print(f"  值: {unique_norms_sq}\n")

# 成对距离分析（采样）
print("成对距离分析（采样前100对）:")
sample_size = min(100, n_points)
sample_idx = np.random.choice(n_points, sample_size, replace=False)
sample_points = sphere_centers[sample_idx]

distances_sq = []
for i in range(len(sample_points)):
    for j in range(i+1, len(sample_points)):
        diff = sample_points[i] - sample_points[j]
        dist_sq = np.sum(diff**2)
        distances_sq.append(dist_sq)

distances_sq = np.array(distances_sq)
distances = np.sqrt(distances_sq)

print(f"  最小距离: {distances.min():.2f}")
print(f"  最大距离: {distances.max():.2f}")
print(f"  平均距离: {distances.mean():.2f}")
print(f"  距离平方唯一值数（采样）: {len(np.unique(distances_sq))}\n")

# 检查对称性（是否有相反点）
print("对称性检查:")
has_opposite = 0
for i in range(min(50, n_points)):
    opposite = -sphere_centers[i]
    for j in range(n_points):
        if np.allclose(opposite, sphere_centers[j], rtol=1e-9):
            has_opposite += 1
            break
print(f"  前50个点中有相反点的数量: {has_opposite}\n")

# 坐标统计
print("坐标值统计:")
coords_flat = sphere_centers.flatten()
print(f"  坐标范围: [{coords_flat.min()}, {coords_flat.max()}]")
print(f"  坐标均值: {coords_flat.mean():.2f}")
print(f"  坐标标准差: {coords_flat.std():.2e}")
print(f"  零坐标数: {np.sum(coords_flat == 0)}/{coords_flat.size}\n")

# 检查整数性质（GCD）
print("整数性质:")
all_coords = sphere_centers.flatten().astype(np.int64)
from math import gcd
from functools import reduce
coords_gcd = reduce(gcd, map(abs, all_coords[all_coords != 0]))
print(f"  所有非零坐标的GCD: {coords_gcd}")
if coords_gcd > 1:
    print(f"  可简化：所有坐标可除以 {coords_gcd}\n")
else:
    print(f"  已简化到最小整数形式\n")

# 约束验证
print("约束验证:")
print(f"  min(成对距离) >= max(范数): {distances.min():.2f} >= {norms.max():.2f}")
if distances.min() >= norms.max():
    print("   满足 kissing number 约束")
    margin = distances.min() - norms.max()
    print(f"  约束余量: {margin:.2f} ({margin/norms.max()*100:.2f}%)")
else:
    print("   不满足约束")

print("\n=== 构造特点推断 ===")
if len(np.unique(norms_squared)) == 1:
    print(" 统一范数  可能使用球面码或格点构造")
if has_opposite > 40:
    print(" 高度中心对称  可能基于 1 设计或对称群")
if coords_gcd > 1000000:
    print(f" 大公约数 {coords_gcd}  可能是标度放大的小整数格")
unique_dist_ratio = len(np.unique(distances_sq)) / len(distances_sq)
if unique_dist_ratio < 0.1:
    print(f" 距离高度离散（唯一值比例 {unique_dist_ratio:.1%}） 规则格结构")

