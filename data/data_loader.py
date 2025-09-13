from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

class UnifiedTimeSeriesDataset(Dataset):
    """
    统一时间序列数据集加载器
    功能特性：
    1. 自动去除首列时间戳
    2. 线性插值处理缺失值
    3. 训练集驱动的归一化
    4. 严格时序划分防止数据泄露
    """

    def __init__(self, data_path, flag='train', seq_len=24, pred_len=12, 
                 train_ratio=0.7, val_ratio=0.1, scale=True, scaler=None):
        """
        参数说明：
        data_path: 数据文件路径
        flag: 数据集类型 ['train', 'val', 'test']
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        scale: 是否进行归一化
        scaler: 预训练归一化器（用于验证/测试集）
        """
        assert flag in ['train', 'val', 'test'], "非法数据集类型"

        # 1. 数据加载与预处理
        # 读取原始数据（假设第一列为时间戳）
        feature_df = pd.read_csv(data_path)
        
        # # (如首列为时间列)删除首列时间戳，保留特征列
        # feature_df = raw_df.iloc[:, 1:]  # 去除第一列
        
        # 处理缺失值：线性插值 -> 前向填充 -> 后向填充，移除全0列
        processed_df = feature_df.interpolate(method='linear', axis=0).ffill().bfill()
        zero_columns = processed_df.columns[(processed_df == 0).all()]
        processed_df = processed_df.drop(columns=zero_columns)
        full_data = processed_df.values  

        # 2. 时序划分（严格按时间顺序）
        total_samples = len(full_data)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        # 根据数据集类型截取数据段
        if flag == 'train':
            self.data = full_data[:train_end]
        elif flag == 'val':
            self.data = full_data[train_end:val_end]
        else:
            self.data = full_data[val_end:]


        # 3. 归一化处理
        self.scale = scale
        if scale:
            if flag == 'train':
                # 训练集：创建并拟合归一化器
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.data = self.scaler.fit_transform(self.data)
            else:
                # 验证/测试集：使用训练集的归一化器
                assert scaler is not None, "验证/测试集必须提供归一化器"
                self.scaler = scaler
                self.data = self.scaler.transform(self.data)
        else:
            self.scaler = None


        # 4. 序列参数设置
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        """生成输入-输出序列对"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end].astype(np.float32)  # 输入序列
        seq_y = self.data[r_begin:r_end].astype(np.float32)  # 目标序列
        return seq_x, seq_y

    def __len__(self):
        """有效序列数量（防止越界）"""
        return len(self.data) - self.seq_len - self.pred_len + 1
    