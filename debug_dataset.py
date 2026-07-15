import os
import traceback
import numpy as np
import cv2
import random
from dataset_augmented_simulation import get_augmented_dataloader

# 创建保存调试图像和轨迹的文件夹
os.makedirs("debug_maps", exist_ok=True)
os.makedirs("debug_tracks", exist_ok=True)
os.makedirs("debug_failed_tracks", exist_ok=True)

num_passes = 20  # 运行轮数
samples_per_pass = 100  # 每轮样本数

print("🔍 开始多轮 Dataset 调试...\n")

for pass_id in range(num_passes):
    seed = random.randint(0, 99999)
    print(f"\n🔁 第 {pass_id+1}/{num_passes} 次，随机种子 {seed}")
    
    train_loader, _, _ = get_augmented_dataloader(
        data_length=samples_per_pass,
        seed=seed,
        batch_size=1,
        debug=True
    )

    for i, batch in enumerate(train_loader):
        sample_id = f"pass{pass_id}_sample{i}"
        try:
            scen_map = batch["scen_map"][0].numpy()  # shape: (3, H, W)
            scen_map = (scen_map * 255).astype("uint8")
            assert scen_map.shape[1] > 0 and scen_map.shape[2] > 0, f"{sample_id} 场景图为空"
            assert not np.isnan(scen_map).any(), f"{sample_id} 场景图中有 NaN"
            assert not np.isinf(scen_map).any(), f"{sample_id} 场景图中有 Inf"
            if scen_map.shape[0] == 3:
                scen_map = scen_map.transpose(1, 2, 0)
            cv2.imwrite(f"debug_maps/{sample_id}.png", scen_map)

            track = batch["observed_data"][0].numpy()  # shape: (L, 2)
            assert not np.isnan(track).any(), f"{sample_id} 轨迹中有 NaN"
            assert not np.isinf(track).any(), f"{sample_id} 轨迹中有 Inf"
            assert track.ndim == 2 and track.shape[1] == 2, f"{sample_id} 轨迹 shape 错误: {track.shape}"

            np.save(f"debug_tracks/{sample_id}.npy", track)

        except Exception as e:
            print(f"[✗] {sample_id} 出错: {e}")
            traceback.print_exc()
            try:
                # 尝试保存失败样本
                np.save(f"debug_failed_tracks/{sample_id}_track.npy", batch["observed_data"][0].numpy())
                scen_map = batch["scen_map"][0].numpy()
                if scen_map.shape[0] == 3:
                    scen_map = scen_map.transpose(1, 2, 0)
                scen_map = (scen_map * 255).astype("uint8")
                cv2.imwrite(f"debug_failed_tracks/{sample_id}_map.png", scen_map)
            except Exception as e2:
                print(f"[!] 无法保存失败样本 {sample_id} 的内容: {e2}")

print("\n✅ Dataset debug 完成。输出保存在 debug_maps/、debug_tracks/ 和 debug_failed_tracks/ 文件夹中。")