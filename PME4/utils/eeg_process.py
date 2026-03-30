
# 生成优化后的Python脚本

# -*- coding: utf-8 -*-
"""
PME4 EEG数据处理脚本
功能：对原始EEG数据进行带通滤波(0.5-45Hz)和降采样(5kHz->1kHz)
输出文件名格式: s01_t001_processed_eeg_1kHz_0.5_45.npy

使用方法:
    python process_pme4_eeg.py
    python process_pme4_eeg.py --base_dir /path/to/PME4
    python process_pme4_eeg.py --lowcut 1 --highcut 50
"""

import numpy as np
import os
from scipy import signal
import time
import argparse


def process_eeg_file(input_path, output_path, lowcut=0.5, highcut=45, fs_original=5000, fs_target=1000):
    """
    处理单个EEG文件：带通滤波 + 降采样
    
    处理流程:
    1. 加载原始EEG数据 (5kHz, 8通道)
    2. 带通滤波 (Butterworth 4阶, 零相位)
    3. 降采样到1kHz (使用decimate, 包含抗混叠滤波)
    4. 保存处理后的数据
    """
    try:
        # 加载原始EEG数据
        eeg_data = np.load(input_path)
        
        # 确保数据是2D (channels, samples)
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        n_channels, n_samples = eeg_data.shape
        
        # 1. 带通滤波 (Butterworth滤波器, 4阶, 零相位)
        nyquist = fs_original / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # 使用SOS (Second-Order Sections) 格式提高数值稳定性
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered_data = signal.sosfiltfilt(sos, eeg_data, axis=1)
        
        # 2. 降采样到1000Hz
        # 降采样因子 = 5000 / 1000 = 5
        decimation_factor = fs_original // fs_target
        
        # 使用scipy.signal.decimate进行降采样
        # 默认使用Chebyshev IIR滤波器 (8阶), 包含抗混叠保护
        # zero_phase=True 确保零相位失真
        downsampled_data = np.zeros((n_channels, n_samples // decimation_factor))
        for i in range(n_channels):
            downsampled_data[i] = signal.decimate(
                filtered_data[i], 
                decimation_factor, 
                ftype='iir',      # IIR滤波器, 计算效率高
                axis=0,
                zero_phase=True   # 零相位滤波
            )
        
        # 保存处理后的数据
        np.save(output_path, downsampled_data)
        
        return True, f"成功: {os.path.basename(input_path)}"
        
    except Exception as e:
        return False, f"错误: {input_path} - {str(e)}"


def batch_process_pme4(base_dir='/data2/zhiwen/dataset/PME4', 
                       lowcut=0.5, 
                       highcut=45, 
                       fs_original=5000, 
                       fs_target=1000,
                       skip_existing=True):
    """
    批量处理PME4数据集中的所有EEG文件
    
    遍历结构：
        /data2/zhiwen/dataset/PME4/
        ├── s01/
        │   ├── t001/
        │   │   └── s01_t001_raw_eeg_5kHz.npy  -> s01_t001_processed_eeg_1kHz_0.5_45.npy
        │   ├── t002/
        │   │   └── s01_t002_raw_eeg_5kHz.npy  -> s01_t002_processed_eeg_1kHz_0.5_45.npy
        │   └── ... (t001-t350)
        ├── s02/
        │   └── ...
        └── s11/
            └── ...
    """
    
    # 统计信息
    total_files = 0
    success_count = 0
    error_count = 0
    skipped_count = 0
    error_files = []
    
    start_time = time.time()
    
    print("=" * 70)
    print("PME4 EEG 批处理脚本")
    print("=" * 70)
    print(f"基础目录: {base_dir}")
    print(f"滤波范围: {lowcut}-{highcut} Hz (Butterworth 4阶, 零相位)")
    print(f"降采样: {fs_original}Hz -> {fs_target}Hz (降采样因子={fs_original//fs_target})")
    print(f"输出格式: [subject]_[trial]_processed_eeg_1kHz_{lowcut}_{highcut}.npy")
    print(f"跳过已存在: {skip_existing}")
    print("=" * 70)
    
    # 遍历所有受试者 s01 到 s11
    for subject_id in range(1, 12):
        subject = f"s{subject_id:02d}"  # s01, s02, ..., s11
        subject_path = os.path.join(base_dir, subject)
        
        # 检查受试者文件夹是否存在
        if not os.path.exists(subject_path):
            print(f"\\n警告: 受试者文件夹不存在: {subject_path}")
            continue
        
        print(f"\\n处理受试者: {subject}")
        subject_start = time.time()
        subject_files = 0
        subject_success = 0
        
        # 遍历所有试次 t001 到 t350
        for trial_id in range(1, 351):
            trial = f"t{trial_id:03d}"  # t001, t002, ..., t350
            trial_path = os.path.join(subject_path, trial)
            
            # 检查试次文件夹是否存在
            if not os.path.exists(trial_path):
                continue
            
            # 构建原始文件名: s01_t001_raw_eeg_5kHz.npy
            raw_filename = f"{subject}_{trial}_raw_eeg_5kHz.npy"
            raw_filepath = os.path.join(trial_path, raw_filename)
            
            # 检查原始文件是否存在
            if not os.path.exists(raw_filepath):
                continue
            
            # 构建输出文件名: s01_t001_processed_eeg_1kHz_0.5_45.npy
            output_filename = f"{subject}_{trial}_processed_eeg_1kHz_{lowcut}_{highcut}.npy"
            output_filepath = os.path.join(trial_path, output_filename)
            
            total_files += 1
            subject_files += 1
            
            # 如果输出文件已存在且设置了跳过，则跳过
            if skip_existing and os.path.exists(output_filepath):
                skipped_count += 1
                success_count += 1
                continue
            
            # 处理文件
            success, msg = process_eeg_file(
                raw_filepath, 
                output_filepath, 
                lowcut=lowcut, 
                highcut=highcut,
                fs_original=fs_original,
                fs_target=fs_target
            )
            
            if success:
                success_count += 1
                subject_success += 1
            else:
                error_count += 1
                error_files.append((raw_filepath, msg))
                print(f"  错误: {msg}")
        
        subject_time = time.time() - subject_start
        if subject_files > 0:
            print(f"  {subject} 完成: {subject_files} 个文件, 成功 {subject_success}, 耗时 {subject_time:.1f} 秒")
    
    # 打印统计信息
    total_time = time.time() - start_time
    print("\\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"总文件数: {total_files}")
    print(f"成功处理: {success_count - skipped_count}")
    print(f"跳过(已存在): {skipped_count}")
    print(f"失败: {error_count}")
    print(f"总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    
    if error_files:
        print("\\n失败的文件列表:")
        for filepath, error_msg in error_files[:10]:
            print(f"  - {filepath}")
            print(f"    {error_msg}")
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files) - 10} 个错误未显示")
    
    print("=" * 70)
    return success_count, error_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='PME4 EEG数据处理脚本 - 带通滤波和降采样',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数处理
  python process_pme4_eeg.py
  
  # 指定数据目录
  python process_pme4_eeg.py --base_dir /data2/zhiwen/dataset/PME4
  
  # 修改滤波范围 (例如1-50Hz)
  python process_pme4_eeg.py --lowcut 1 --highcut 50
  
  # 强制重新处理已存在的文件
  python process_pme4_eeg.py --no-skip
  
  # 组合使用
  python process_pme4_eeg.py --base_dir /path/to/PME4 --lowcut 0.1 --highcut 100
        """
    )
    
    parser.add_argument('--base_dir', type=str, default='/data2/zhiwen/dataset/PME4',
                        help='PME4数据集根目录 (默认: /data2/zhiwen/dataset/PME4)')
    parser.add_argument('--lowcut', type=float, default=0.5,
                        help='低截止频率 (Hz) (默认: 0.5)')
    parser.add_argument('--highcut', type=float, default=45,
                        help='高截止频率 (Hz) (默认: 45)')
    parser.add_argument('--fs_original', type=int, default=5000,
                        help='原始采样率 (Hz) (默认: 5000)')
    parser.add_argument('--fs_target', type=int, default=1000,
                        help='目标采样率 (Hz) (默认: 1000)')
    parser.add_argument('--no-skip', action='store_true',
                        help='不跳过已存在的输出文件，强制重新处理')
    
    args = parser.parse_args()
    
    # 执行批处理
    success, errors = batch_process_pme4(
        base_dir=args.base_dir,
        lowcut=args.lowcut,
        highcut=args.highcut,
        fs_original=args.fs_original,
        fs_target=args.fs_target,
        skip_existing=not args.no_skip
    )
    
    # 返回退出码
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)


# 保存到文件
output_path = '/mnt/kimi/output/process_pme4_eeg.py'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(script_content)

print(f"✅ 脚本已保存到: {output_path}")
print(f"📄 文件大小: {len(script_content)} 字符")
print("\\n" + "="*70)
print("脚本功能说明:")
print("="*70)
print("1. 遍历 /data2/zhiwen/dataset/PME4 下的 s01-s11 文件夹")
print("2. 对每个文件夹下的 t001-t350 进行遍历")
print("3. 读取 s01_t001_raw_eeg_5kHz.npy 等原始EEG文件")
print("4. 进行带通滤波 (0.5-45Hz, Butterworth 4阶, 零相位)")
print("5. 降采样到 1000Hz (使用 scipy.signal.decimate)")
print("6. 保存为 s01_t001_processed_eeg_1kHz_0.5_45.npy")
print("="*70)
