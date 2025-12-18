import numpy as np
import sys
from collections import Counter
sys.path.insert(0, "examples/kissing_number_11")
from sota import sphere_centers

points = sphere_centers.astype(np.float64)
n_points = len(points)

print("=== 深度构造分析 ===\n")

# 1. 精确范数分析
norms_sq = np.sum(points**2, axis=1)
unique_norms_sq = np.unique(norms_sq)
print(f"1. 范数平方的唯一值:")
print(f"   数量: {len(unique_norms_sq)}")
if len(unique_norms_sq) <= 10:
    for i, ns in enumerate(unique_norms_sq):
        count = np.sum(norms_sq == ns)
        print(f"   [{i}] {ns:.6e} ({count} 个点)")

# 找出最常见的范数
norm_counter = Counter(norms_sq)
most_common_norm_sq = norm_counter.most_common(1)[0]
print(f"\n   最常见范数平方: {most_common_norm_sq[0]:.6e} ({most_common_norm_sq[1]}/{n_points} = {most_common_norm_sq[1]/n_points*100:.1f}%)")

# 2. 标度因子推断
print(f"\n2. 标度因子分析:")
# 计算平均范数
avg_norm = np.sqrt(most_common_norm_sq[0])
print(f"   主要范数: {avg_norm:.6e}")

# 尝试找小整数基准
suspected_bases = [10, 100, 1000, 10000]
for base in suspected_bases:
    ratio = avg_norm / (base**6)
    if 0.9 < ratio < 1.1:
        print(f"   可能基于: {base}^6 (ratio={ratio:.6f})")

# 3. 成对距离平方的离散值分析
print(f"\n3. 成对距离平方的离散模式:")
dist_sq_samples = []
sample_size = min(150, n_points)
for i in range(sample_size):
    for j in range(i+1, sample_size):
        diff = points[i] - points[j]
        dist_sq = np.sum(diff**2)
        dist_sq_samples.append(dist_sq)

dist_sq_samples = np.array(dist_sq_samples)
unique_dist_sq = np.unique(dist_sq_samples)
print(f"   唯一距离平方值数: {len(unique_dist_sq)} (采样 {len(dist_sq_samples)} 对)")
print(f"   离散度: {len(unique_dist_sq)/len(dist_sq_samples)*100:.1f}%")

# 找最小距离
min_dist_sq = unique_dist_sq[0]
print(f"   最小距离平方: {min_dist_sq:.6e}")
print(f"   最小距离: {np.sqrt(min_dist_sq):.6e}")

# 检查距离是否是某个基数的倍数
dist_ratios = unique_dist_sq[:10] / min_dist_sq
print(f"\n   前10个距离平方与最小值的比值:")
for i, ratio in enumerate(dist_ratios):
    print(f"   [{i}] {ratio:.6f}")

# 4. 坐标值的离散化特征
print(f"\n4. 坐标值分布:")
coords_flat = points.flatten()
unique_coords = np.unique(coords_flat)
print(f"   唯一坐标值数: {len(unique_coords)}/{len(coords_flat)}")

# 检查坐标是否集中在某些值
coord_hist, _ = np.histogram(coords_flat, bins=50)
print(f"   最密集bin的占比: {coord_hist.max()/len(coords_flat)*100:.2f}%")

# 5. 内积分析（角度分布）
print(f"\n5. 点间角度特征:")
sample_inner_products = []
for i in range(min(50, n_points)):
    for j in range(i+1, min(50, n_points)):
        inner_prod = np.dot(points[i], points[j])
        # 归一化内积 = cos(theta)
        cos_theta = inner_prod / (np.linalg.norm(points[i]) * np.linalg.norm(points[j]))
        sample_inner_products.append(cos_theta)

sample_inner_products = np.array(sample_inner_products)
print(f"   cos(theta) 范围: [{sample_inner_products.min():.6f}, {sample_inner_products.max():.6f}]")
print(f"   cos(theta) 均值: {sample_inner_products.mean():.6f}")

# 检查是否有特殊角度
special_angles = [0, 0.5, -0.5, 1/np.sqrt(2), -1/np.sqrt(2)]
for angle in special_angles:
    close_count = np.sum(np.abs(sample_inner_products - angle) < 0.01)
    if close_count > 0:
        print(f"   接近 cos({angle:.4f}) 的角度数: {close_count}")

print("\n=== 推断的构造策略 ===")
print(f" 球面码: 所有点在 R={avg_norm:.3e} 的球面上")
print(f" 饱和配置: min_dist  R (接触条件)")
print(f" 中心对称: 质心接近原点")
print(f" 各向同性: 特征值比 ~1.36")
print(f"\n可能的构造方法:")
print(f"   基于格的构造（如 E8, Leech lattice 在 R^11 的投影）")
print(f"   球面设计或组合优化")
print(f"   点数 593 接近理论上界，可能是已知的最优或近优配置")

