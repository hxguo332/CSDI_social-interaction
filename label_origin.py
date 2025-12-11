import matplotlib.pyplot as plt
import matplotlib.image as mpimg

i = 4
png = f"dataset_RTLS/german_{i}/german_{i}_full.png"  # 换成你的 PNG
rough = None  # 粗略坐标(如已有估计), 例如 (2128, 1516)

img = mpimg.imread(png)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
ax.set_title("工具栏放大/拖拽，满意后按 Enter，再点击原点", fontsize=12)

# 如果有粗略估计，先预缩放到附近，方便二次放大
if rough:
    x0, y0 = rough
    pad = 200  # 预留窗口大小
    ax.set_xlim(x0 - pad, x0 + pad)
    ax.set_ylim(y0 + pad, y0 - pad)  # 注意 y 轴向下

plt.tight_layout()
plt.show(block=False)

input("在图上用工具栏放大/拖拽，定位到原点附近，满意后按 Enter...")
pts = plt.ginput(1, timeout=-1)  # 点击一次原点
plt.close(fig)
print("clicked:", pts)

# origin for german_1: (314, 1037)
# origin for german_2: (629, 404)
# origin for german_3: (515, 754)
# origin for german_4: (160, 272)