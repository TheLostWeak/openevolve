import numpy as np
import sys
sys.path.insert(0, "examples/kissing_number_11")
from sota import sphere_centers

print("=== SOTA 解分析（R^11 Kissing Number）===\n")

# 转换为 float64 避免整数溢出
points = sphere_centers.astype(np.float64)
n_points = len(points)
dim = points.shape[1]

print(f"点数: {n_points}")
print(f"维度: {dim}\n")

# 范数统计
norms_squared = np.sum(points**2, axis=1)
norms = np.sqrt(norms_squared)
print(f"范数统计:")
print(f"  最小范数: {norms.min():.6e}")
print(f"  最大范数: {norms.max():.6e}")
print(f"  平均范数: {norms.mean():.6e}")
print(f"  范数标准差: {norms.std():.6e}")
print(f"  范数变异系数: {norms.std()/norms.mean():.6f}\n")

# 检查范数是否近似统一
norm_cv = norms.std() / norms.mean()
if norm_cv < 0.01:
    print(" 范数高度统一（变异系数 < 1%）")
elif norm_cv < 0.1:
    print(" 范数接近统一（变异系数 < 10%）")
else:
    print(" 范数差异较大")

# 成对距离完整计算（子集）
print("\n成对距离分析:")
min_dist = float("inf")
max_dist = 0
dist_samples = []

# 计算每个点到其他所有点的最小距离
for i in range(min(200, n_points)):
    for j in range(i+1, min(200, n_points)):
        diff = points[i] - points[j]
        dist = np.sqrt(np.sum(diff**2))
        dist_samples.append(dist)
        if dist < min_dist:
            min_dist = dist
        if dist > max_dist:
            max_dist = dist

dist_samples = np.array(dist_samples)
print(f"  最小距离: {min_dist:.6e}")
print(f"  最大距离: {max_dist:.6e}")
print(f"  平均距离: {dist_samples.mean():.6e}")
print(f"  距离标准差: {dist_samples.std():.6e}\n")

# 约束验证
print("约束验证 (min_pairwise >= max_norm):")
print(f"  min(成对距离): {min_dist:.6e}")
print(f"  max(范数): {norms.max():.6e}")
ratio = min_dist / norms.max()
print(f"  比值: {ratio:.6f}")
if ratio >= 1.0:
    print(f"   满足约束（余量 {(ratio-1)*100:.4f}%）")
else:
    print(f"   不满足约束（缺口 {(1-ratio)*100:.4f}%）")

# 坐标统计
print("\n坐标值统计:")
coords_flat = points.flatten()
print(f"  坐标范围: [{coords_flat.min():.6e}, {coords_flat.max():.6e}]")
print(f"  坐标绝对值均值: {np.abs(coords_flat).mean():.6e}")
print(f"  零坐标占比: {np.sum(np.abs(coords_flat) < 1)/coords_flat.size*100:.2f}%")

# 对称性检查
print("\n对称性分析:")
center = points.mean(axis=0)
print(f"  质心偏移: {np.linalg.norm(center):.6e}")
if np.linalg.norm(center) < norms.mean() * 0.01:
    print("   近似中心对称（质心接近原点）")
else:
    print("   质心偏离原点")

# 检查是否有近似相反的点
opposite_pairs = 0
for i in range(min(100, n_points)):
    opposite = -points[i]
    dists_to_opposite = np.sqrt(np.sum((points - opposite)**2, axis=1))
    if np.min(dists_to_opposite) < norms.mean() * 0.01:
        opposite_pairs += 1

print(f"  相反点对数（前100检测）: {opposite_pairs}")

# 特征值分析（协方差）
print("\n协方差矩阵特征值分析:")
cov = np.cov(points.T)
eigenvalues = np.linalg.eigvalsh(cov)
eigenvalues = np.sort(eigenvalues)[::-1]
print(f"  最大特征值: {eigenvalues[0]:.6e}")
print(f"  最小特征值: {eigenvalues[-1]:.6e}")
print(f"  特征值比（各向异性）: {eigenvalues[0]/eigenvalues[-1]:.2f}")
if eigenvalues[0]/eigenvalues[-1] < 2:
    print("   接近各向同性（球对称）")
else:
    print("   有方向偏好")

print("\n=== 构造特点总结 ===")
print(f" 点集大小: {n_points}（R^11 kissing number 下界约 438-582）")
print(f" 范数变异: {norm_cv:.4f}  ", end="")
if norm_cv < 0.01:
    print("球面码构造")
elif norm_cv < 0.1:
    print("准球面码")
else:
    print("多壳层结构")

print(f" 约束余量: {(ratio-1)*100:.4f}%  ", end="")
if ratio > 1.0001:
    print("有优化空间")
elif ratio > 0.9999:
    print("几乎饱和")
else:
    print("约束违反")

if opposite_pairs > 80:
    print(" 高度中心对称  可能基于格/群构造")
elif opposite_pairs > 20:
    print(" 部分对称性")
else:
    print(" 低对称性  可能是数值优化结果")

