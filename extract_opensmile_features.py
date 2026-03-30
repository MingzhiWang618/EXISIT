import os
import pickle
import numpy as np
import opensmile
from tqdm import tqdm
from scipy import signal
import traceback  # 引入堆栈打印工具

def process_subject_audio_data(subject_id, input_dir, output_dir, smile_extractor):
    """
    处理单个被试音频：提取 LLD -> 重采样 32 帧 -> 直接保存原始特征值
    """
    input_file = os.path.join(input_dir, f"subject_{subject_id:02d}_aud.pkl")
    output_file = os.path.join(output_dir, f"subject_{subject_id:02d}_aud_egemaps_32f_raw.pkl")
    
    if not os.path.exists(input_file):
        print(f"❌ 跳过：找不到文件 {input_file}")
        return

    print(f"\n🚀 正在处理被试 {subject_id:02d}...")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    train_x, train_y, test_x, test_y = data

    def extract_and_resample(waveform_list, desc):
        processed_features = []
        error_count = 0
        
        # 使用 tqdm 包装循环
        pbar = tqdm(waveform_list, desc=desc, leave=False)
        for i, x in enumerate(pbar):
            try:
                # 确保输入形状为 [1, samples]
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                
                # 1. 提取特征
                # 注意：如果采样率不是 16000，请确保这里传入正确的 fs
                df = smile_extractor.process_signal(x, 16000)
                lld_values = np.nan_to_num(df.values, nan=0.0)
                
                # 2. 时间轴重采样
                if lld_values.shape[0] >= 2: # 至少要有两帧才能重采样
                    resampled = signal.resample(lld_values, 32, axis=0)
                else:
                    # 如果只有1帧或0帧，记录警告
                    # print(f"\n⚠️ 警告: 第 {i} 个样本太短 (帧数: {len(lld_values)})")
                    resampled = np.zeros((32, 25))
                
                processed_features.append(resampled.astype(np.float32))
                
            except Exception as e:
                error_count += 1
                # 打印详细错误信息
                print(f"\n--- ❌ 提取错误 (样本索引: {i}) ---")
                print(f"错误类型: {type(e).__name__}")
                print(f"具体原因: {e}")
                # 如果你想看更详细的报错代码行数，取消下面这行的注释:
                # traceback.print_exc() 
                
                processed_features.append(np.zeros((32, 25), dtype=np.float32))
        
        if error_count > 0:
            print(f"\nℹ️ {desc} 完成，共出现 {error_count} 处错误。")
            
        return np.array(processed_features)

    # 分别处理
    new_train_x = extract_and_resample(train_x, "训练集提取")
    new_test_x = extract_and_resample(test_x, "测试集提取")

    # 保存
    with open(output_file, 'wb') as f:
        pickle.dump([new_train_x, train_y, new_test_x, test_y], f)
    
    print(f"✅ 处理完成。特征形状: {new_train_x.shape}")

def batch_process(input_dir, output_dir, start_id=1, end_id=42):
    os.makedirs(output_dir, exist_ok=True)
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    for sid in range(start_id, end_id + 1):
        process_subject_audio_data(sid, input_dir, output_dir, smile)

if __name__ == "__main__":
    INPUT_DIR = "/data2/mingzhi/BCI/dataset/EAV_old/Audio"
    OUTPUT_DIR = "/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw"

    batch_process(INPUT_DIR, OUTPUT_DIR, 1, 42)
    print("\n✨ 任务结束。")