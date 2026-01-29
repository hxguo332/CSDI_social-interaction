import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

png = Path("dataset_RTLS/german_4/german_4_full.png")  # 换成你的 PNG
meter_label = 18  # 标尺标注的米数
patch_size = 200  # 裁剪窗口半径，可调

def pick_point(img, title):
    plt.imshow(img)
    plt.title(title)
    pt = plt.ginput(1, timeout=-1)[0]
    plt.close()
    return pt

def zoom_pick(png_path, rough_pt):
    img = mpimg.imread(png_path)
    cx, cy = rough_pt
    x0 = max(int(cx - patch_size), 0)
    y0 = max(int(cy - patch_size), 0)
    x1 = min(int(cx + patch_size), img.shape[1])
    y1 = min(int(cy + patch_size), img.shape[0])
    crop = Image.open(png_path).crop((x0, y0, x1, y1)).resize((800, 800), resample=Image.NEAREST)
    plt.imshow(crop)
    plt.title("在放大图中精确点选")
    xz, yz = plt.ginput(1, timeout=-1)[0]
    plt.close()
    # 还原到原图坐标
    x_orig = x0 + xz * ((x1 - x0) / 800.0)
    y_orig = y0 + yz * ((y1 - y0) / 800.0)
    return (x_orig, y_orig)

# 第一步：粗选起点
img_full = mpimg.imread(png)
p1_rough = pick_point(img_full, "全图粗选起点（标尺一端）")
p1 = zoom_pick(png, p1_rough)

# 第二步：粗选终点
p2_rough = pick_point(img_full, "全图粗选终点（标尺另一端）")
p2 = zoom_pick(png, p2_rough)

px_len = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
scale_px_per_m = px_len / meter_label

print("起点(精):", p1)
print("终点(精):", p2)
print(f"像素长度: {px_len:.2f} px")
print(f"比例: {scale_px_per_m:.4f} px/米")


# german_1: 44.56 米宽
# 起点(精): (np.float64(314.37337662337666), np.float64(940.672077922078))
# 终点(精): (np.float64(2840.1612554112553), np.float64(939.3647186147186))
# 像素长度: 2525.79 px
# 比例: 56.6829 px/米

# german_3: 127 米长
# 起点(精): (np.float64(526.5725108225109), np.float64(783.2348484848485))
# 终点(精): (np.float64(2329.3603896103896), np.float64(783.871212121212))
# 像素长度: 1802.79 px
# 比例: 14.1952 px/米

# german_4: 18 米长
# 起点(精): (np.float64(184.48727272727277), np.float64(311.788961038961))
# 终点(精): (np.float64(839.9967532467533), np.float64(310.90584415584414))
# 像素长度: 655.51 px
# 比例: 36.4172 px/米