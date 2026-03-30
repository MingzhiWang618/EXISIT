import os
import subprocess
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import pandas as pd
import shutil
import atexit
import warnings
import traceback

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
OPENFACE_PATH = "/data2/mingzhi/BCI/WMZ_BCI/OpenFace-master/build/bin/FeatureExtraction"
INPUT_DIR     = "/data2/mingzhi/BCI/dataset/EAV_old/Vision1"
OUTPUT_DIR    = "/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace"

IMG_W, IMG_H  = 224, 224
JPEG_QUALITY  = 95


# ─────────────────────────────────────────────
# 归一化
# ─────────────────────────────────────────────
def normalize_features(arr: np.ndarray, col_names: list) -> np.ndarray:
    arr = arr.astype(np.float32).copy()

    for i, col in enumerate(col_names):
        c = col.strip()

        if c.startswith('x_'):
            arr[:, i] = np.clip(arr[:, i], 0, IMG_W) / IMG_W

        elif c.startswith('y_'):
            arr[:, i] = np.clip(arr[:, i], 0, IMG_H) / IMG_H

        elif c.startswith('AU') and c.endswith('_r'):
            arr[:, i] = np.clip(arr[:, i], 0, 5) / 5.0

        elif c.startswith('gaze_'):
            arr[:, i] = np.clip(arr[:, i], -1, 1)

    return arr


# ─────────────────────────────────────────────
# 单 trial 提取（单线程 + 打印所有错误）
# ─────────────────────────────────────────────
def extract_one_trial(trial_frames, trial_unique_id, openface_path, tmp_base):

    trial_dir  = os.path.join(tmp_base, f"trial_{trial_unique_id}")
    frames_dir = os.path.join(trial_dir, "frames")
    out_dir    = os.path.join(trial_dir, "out")

    try:
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        n = len(trial_frames)
        if n == 0:
            print(f"⚠ 空 Trial: {trial_unique_id}")
            return None

        print(f"\n▶ 正在处理 Trial: {trial_unique_id}")

        # Step1: 写帧
        for i, frame in enumerate(trial_frames):

            img = np.array(frame)

            if img.dtype in [np.float32, np.float64]:
                if img.max() <= 1.0:
                    img = img * 255

            img = img.astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{i:06d}.jpg"),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

        # Step2: 运行 OpenFace
        cmd = [
            openface_path,
            '-fdir', frames_dir,
            '-out_dir', out_dir,
            '-aus',
            '-2Dfp',
            '-gaze',
            '-noimgout',
            '-nomask'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"\n❌ OpenFace 失败: {trial_unique_id}")
            print("STDERR:")
            print(result.stderr)
            return None

        # Step3: 解析 CSV
        csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"\n❌ 未生成 CSV: {trial_unique_id}")
            return None

        df = pd.read_csv(os.path.join(out_dir, csv_files[0]))
        df.columns = [c.strip() for c in df.columns]

        if 'frame' not in df.columns:
            df['frame'] = range(1, len(df) + 1)

        feat_cols = [
            c for c in df.columns
            if (c.startswith('AU') and c.endswith('_r'))
            or c.startswith('gaze_')
            or c.startswith('x_')
            or c.startswith('y_')
        ]

        if not feat_cols:
            print(f"\n❌ 未找到特征列: {trial_unique_id}")
            print("CSV 列名如下:")
            print(df.columns.tolist())
            return None

        # 失败帧处理
        if 'success' in df.columns:
            failed = df['success'].values == 0
            df.loc[failed, feat_cols] = 0.0

        arr = df[feat_cols].values.astype(np.float32)

        # 帧对齐
        if len(arr) != n:
            print(f"⚠ 帧数不一致: OpenFace={len(arr)} 原始={n} | {trial_unique_id}")
            frame_idx = df['frame'].values.astype(int) - 1
            valid_mask = (frame_idx >= 0) & (frame_idx < n)
            aligned = np.zeros((n, len(feat_cols)), dtype=np.float32)
            aligned[frame_idx[valid_mask]] = arr[valid_mask]
            arr = aligned

        return normalize_features(arr, feat_cols)

    except subprocess.TimeoutExpired:
        print(f"\n⏰ OpenFace 超时: {trial_unique_id}")
        return None

    except Exception:
        print(f"\n💥 Python 异常: {trial_unique_id}")
        traceback.print_exc()
        return None

    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# 单 Subject 处理（单线程）
# ─────────────────────────────────────────────
def process_subject(sid, input_pkl, output_pkl, openface_path, tmp_base):

    if not os.path.exists(input_pkl):
        return f"[{sid:02d}] ❌ 找不到输入文件"

    if os.path.exists(output_pkl):
        return f"[{sid:02d}] 跳过: 已存在"

    try:
        with open(input_pkl, 'rb') as f:
            train_x, train_y, test_x, test_y = pickle.load(f)
    except Exception:
        traceback.print_exc()
        return f"[{sid:02d}] ❌ 读取 PKL 失败"

    def run_single(split_x, split_name):

        results = []
        missing_count = 0
        expected_dim = 161

        for i, frames in enumerate(tqdm(split_x, desc=f"Sub{sid:02d}-{split_name}")):

            trial_id = f"{sid}_{split_name}_{i}"
            res = extract_one_trial(frames, trial_id, openface_path, tmp_base)

            if res is not None:
                results.append(res)
                expected_dim = res.shape[1]
            else:
                print(f"⚠ Trial 失败: {trial_id}")
                results.append(np.zeros((len(frames), expected_dim), dtype=np.float32))
                missing_count += 1

        return results, missing_count

    train_feats, m_tr = run_single(train_x, "train")
    test_feats,  m_te = run_single(test_x,  "test")

    with open(output_pkl, 'wb') as f:
        pickle.dump([train_feats, train_y, test_feats, test_y], f)

    return f"[{sid:02d}] ✅ 完成 | Train缺失:{m_tr}/{len(train_x)} Test缺失:{m_te}/{len(test_x)}"


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tmp_base = f"/dev/shm/openface_debug_{os.getpid()}"
    os.makedirs(tmp_base, exist_ok=True)

    atexit.register(lambda: shutil.rmtree(tmp_base, ignore_errors=True))

    print("🚀 单线程 Debug 模式启动")
    print(f"📁 临时路径: {tmp_base}\n")

    for sid in tqdm(range(1, 43), desc="总进度"):

        in_pkl  = os.path.join(INPUT_DIR,  f"subject_{sid:02d}_vis.pkl")
        out_pkl = os.path.join(OUTPUT_DIR, f"subject_{sid:02d}_vis_openface.pkl")

        status = process_subject(sid, in_pkl, out_pkl, OPENFACE_PATH, tmp_base)
        print(status)

    print("\n🎉 全部处理完成！")