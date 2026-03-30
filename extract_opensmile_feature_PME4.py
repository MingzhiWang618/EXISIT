import os
import numpy as np
import opensmile
from tqdm import tqdm
import warnings

# 忽略 OpenSmile 警告
warnings.filterwarnings("ignore", category=UserWarning, module="opensmile")

def extract_pme4_opensmile_window(root_dir):
    """
    从 .wav 中提取 1.0s 到 4.0s 之间的 eGeMAPS LLD 特征。
    """
    # 初始化 OpenSmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    subject_dirs = sorted([d for d in os.listdir(root_dir) 
                           if d.startswith("s") and os.path.isdir(os.path.join(root_dir, d))])

    print(f"📂 正在精准提取 1s-4s 特征...")

    for sub_dir in subject_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        trial_dirs = sorted([d for d in os.listdir(sub_path) 
                             if d.startswith("t") and os.path.isdir(os.path.join(sub_path, d))])
        
        pbar = tqdm(trial_dirs, desc=f"🚀 {sub_dir}", leave=False)
        
        for trial_dir in pbar:
            trial_path = os.path.join(sub_path, trial_dir)
            wav_files = [f for f in os.listdir(trial_path) if f.endswith(".wav")]
            
            for wav_file in wav_files:
                input_wav_path = os.path.join(trial_path, wav_file)
                
                # 输出文件名增加标识 _1s_4s
                output_filename = wav_file.replace(".wav", "_opensmile_egemaps_1s_4s.npy")
                output_filepath = os.path.join(trial_path, output_filename)

                if os.path.exists(output_filepath):
                    continue
                
                try:
                    # --- 核心修改：指定提取的时间窗口 ---
                    # start=1.0 代表从第1秒开始
                    # end=4.0   代表到第4秒结束（总计提取3秒长度）
                    df = smile.process_file(input_wav_path, start=1.0, end=4.0)
                    
                    # 转换为 float32 并处理 NaN
                    lld_values = np.nan_to_num(df.values, nan=0.0).astype(np.float32)
                    
                    # 检查帧数：
                    # eGeMAPS 默认 10ms 一帧，3秒音频预期得到约 299-301 帧
                    # print(f"DEBUG: {wav_file} shape: {lld_values.shape}")
                    
                    np.save(output_filepath, lld_values)
                    
                except Exception as e:
                    print(f"\n❌ 提取失败 {wav_file}: {e}")

if __name__ == "__main__":
    ROOT_DIR = "/data2/zhiwen/dataset/PME4/extracted"
    extract_pme4_opensmile_window(ROOT_DIR)
    print("\n✨ 精准提取任务完成！")