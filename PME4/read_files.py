import os
import numpy as np
import pandas as pd

# 读取EEG数据文件
def read_eeg_data():
    eeg_file_path = "/data2/zhiwen/dataset/PME4/extracted/s01/t134/s01_t134_processed_eeg_1kHz.npy"
    if os.path.exists(eeg_file_path):
        print(f"正在读取EEG数据文件: {eeg_file_path}")
        eeg_data = np.load(eeg_file_path)
        print(f"EEG数据形状: {eeg_data.shape}")
        print(f"EEG数据类型: {eeg_data.dtype}")
        print(f"EEG数据最小值: {np.min(eeg_data)}")
        print(f"EEG数据最大值: {np.max(eeg_data)}")
        print(f"EEG数据均值: {np.mean(eeg_data)}")
        print(f"EEG数据前5个样本: {eeg_data[:5]}")
        return eeg_data
    else:
        print(f"EEG文件不存在: {eeg_file_path}")
        return None

# 读取PME4数据集配置文件
def read_pme4_config():
    config_file_path = "/data2/zhiwen/dataset/PME4/PME4_dataset_configs.csv"
    if os.path.exists(config_file_path):
        print(f"\n正在读取PME4配置文件: {config_file_path}")
        config_data = pd.read_csv(config_file_path)
        print(f"配置文件形状: {config_data.shape}")
        print("配置文件列名:", list(config_data.columns))
        print("配置文件前5行:")
        print(config_data.head())
        return config_data
    else:
        print(f"配置文件不存在: {config_file_path}")
        return None

if __name__ == "__main__":
    eeg_data = read_eeg_data()
    config_data = read_pme4_config()
