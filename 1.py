import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 生成 8*8 的随机相关系数矩阵
np.random.seed(42)
# 生成 8 个变量的数据
data = np.random.rand(100, 8) 
df = pd.DataFrame(data)
corr_matrix = df.corr()

# 2. 设置绘图风格
sns.set_theme(style="white")

# 3. 创建画布
f, ax = plt.subplots(figsize=(6, 6))

# 4. 绘制热力图
sns.heatmap(
    corr_matrix, 
    cmap="YlGnBu",      # 明显的蓝绿黄渐变
    annot=False,       # 不显示数字
    square=True,       # 单元格为正方形
    linewidths=2.5,    # 进一步加粗黑色边界线，适配 8*8 密度
    linecolor='black', # 边界线颜色设为黑色
    cbar=False,        # 去掉图例
    xticklabels=False, # 去掉轴刻度
    yticklabels=False
)

# 5. 彻底移除坐标轴外框
ax.axis('off')

# 保存并显示
plt.tight_layout()
plt.savefig("pcc_matrix_8x8.png", dpi=300, bbox_inches='tight')
plt.show()