# 计算机图形学实验六：可微渲染
## 一、实验目标

通过可微渲染技术，从初始球体优化得到目标牛模型，掌握：
1. 软光栅化的梯度传播原理
2. 网格正则化的作用
3. 多阶段训练策略

---

## 二、实验环境

- **硬件**：阿里云DSW-GPU实例 (A10)
- **软件**：
  - Python 3.11
  - PyTorch 2.9.1 + CUDA 12.8
  - PyTorch3D 0.7.8
  - 分辨率：256×256
  - 视角数量：20

---

## 三、实验过程

### 3.1 必做部分：基于剪影的形状优化

**目标**：仅使用剪影Loss优化形状

**参数配置**：
```python
optimizer = SGD(lr=1.0, momentum=0.9)
epochs = 300
正则化权重：laplacian=1.0, edge=0.1, normal=0.01
```

**结果**：
- 最终Silhouette Loss: 0.001870
- 形状基本贴合，但后期略有震荡
- 迭代步数: 299/300 | 总 Loss: 0.0179 | 剪影误差: 0.0141
<img width="794" height="394" alt="image" src="https://github.com/user-attachments/assets/b4cb0861-7ac9-44de-a142-40eb5c709d26" />

---

### 3.2 改进版1：Adam优化器 + 学习率衰减

**改进点**：
1. 使用Adam优化器（lr=0.01）
2. 余弦退火学习率调度
3. 增加迭代次数至500

**结果**：
- 最终Loss: 0.009967
- 收敛更稳定，无震荡
- 形状更光滑
- Epoch 499/500
- Total Loss:      0.004449
- Silhouette Loss: 0.001038
- Laplacian Loss:  0.003592
- Edge Loss:       0.002787
- Normal Loss:     0.025821
- Learning Rate:   0.000000
  <img width="1040" height="490" alt="image" src="https://github.com/user-attachments/assets/b1d36a21-9add-428f-9172-ffbfd0c7130a" />
<img width="1589" height="490" alt="image" src="https://github.com/user-attachments/assets/fa69e009-f8fd-40ab-a5f1-2e7ab4187e7b" />

---

### 3.3 选做部分：RGB纹理优化（三阶段训练）

#### **核心思想**

传统方法同时优化形状和纹理时，RGB Loss容易主导，导致形状不准确。
采用**三阶段渐进策略**：

**阶段1 (0-600 epoch)**: 纯形状优化
- 只用Silhouette Loss + 正则化
- 专注建立正确的几何结构

**阶段2 (600-900 epoch)**: 形状+纹理联合优化
- 引入RGB Loss (权重1.0)
- Silhouette权重提高至2.0，防止形状退化
- 学习率降至0.005

**阶段3 (900-1100 epoch)**: 精细调整
- 学习率降至0.001
- 每次迭代使用4个视角（原来2个）
- 增强正则化（laplacian=1.5, edge=1.2）

#### **参数对比表**

| 阶段 | Epochs | 学习率 | 优化目标 | RGB权重 | Sil权重 | 视角数 |
|------|--------|--------|----------|---------|---------|--------|
| 1    | 600    | 0.01   | deform_verts | 0 | 1.0 | 2 |
| 2    | 300    | 0.005  | verts + rgb | 1.0 | 2.0 | 2 |
| 3    | 200    | 0.001  | verts + rgb | 1.0 | 2.0 | 4 |

#### **最终结果**

- **Total Loss**: 0.014 (从初始0.27降至0.014)
- **RGB Loss**: 0.003 (颜色高度一致)
- **Silhouette Loss**: 0.002 (形状精准)

<img width="1364" height="1389" alt="image" src="https://github.com/user-attachments/assets/096352f1-d0a9-480e-93b4-1750eb4a1558" />

<img width="1325" height="554" alt="image" src="https://github.com/user-attachments/assets/d86476f7-cebf-4f30-abbe-281720a0b22c" />
从结果可见：
1. 形状完全贴合目标牛的轮廓
2. 光影过渡自然，表面平滑
3. 三个阶段渐进优化，避免了形状退化

---

## 四、关键技术点

### 4.1 软光栅化

使用Sigmoid函数在边界产生平滑梯度：

$$A(d) = \sigma\left(\frac{d}{\text{blur\_radius}}\right)$$

避免了硬光栅化的梯度消失问题。

### 4.2 网格正则化

**拉普拉斯平滑**：
$$
A(d) = \sigma\left(\frac{d}{\text{blur\_radius}}\right)
$$

防止表面出现尖刺和噪声。

**边长一致性**：惩罚过长或过短的边，保持三角形质量。

**法线一致性**：约束相邻面法线方向接近。

### 4.3 多阶段训练

关键洞察：**形状比纹理更重要**
- 先建立正确的几何 → 再添加纹理细节
- 阶段2提高Silhouette权重防止退化
- 阶段3小学习率 + 多视角精修

---

## 五、实验总结

### 5.1 收获

1. **理解了可微渲染的核心**：软光栅化如何提供有效梯度
2. **正则化的重要性**：没有正则化，网格会变成"刺猬"
3. **训练策略比参数更重要**：多阶段训练效果远超单纯调参

### 5.2 改进方向

1. 使用更精细的初始球体（ico_sphere(5)）
2. 增加视角数量至40个
3. 尝试不同的loss权重组合
4. 加入Chamfer Distance作为额外约束

---

## 六、代码仓库

**GitHub**: https://github.com/xiziqi0521/cg-experiment-6

代码结构清晰，包含详细注释，可直接运行复现结果。
