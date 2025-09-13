import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import UnifiedTimeSeriesDataset
from utils.utils import evaluate_metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import numpy as np
from model.SSMGNN import SSMGNN

# 固定随机种子（保证可重复性）
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# 配置参数解析
parser = argparse.ArgumentParser(description='ssm graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='ECG', help='data set')
parser.add_argument('--seq_len', type=int, default=12, help='inout length')
parser.add_argument('--pre_len', type=int, default=12, help='predict length')
parser.add_argument('--batch_size', type=int, default=2, help='input data batch size')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')                   
parser.add_argument('--hidden_size', type=int, default=512, help='hidden dimensions')               
parser.add_argument('--number_frequency', type=int, default=8, help='number of frequency components')
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='optimizer learning rate')
parser.add_argument('--decay_step', type=int, default=5, help='Learning rate decay step')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Learning rate decay rate')
parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
parser.add_argument('--patience', type=int, default=7, help='patience for early stopping')
args = parser.parse_args()
print(f'Training configs: {args}')

# 数据集路径配置
DATA_CONFIG = {
    'traffic': 'data/traffic.csv',
    'ECG': 'data/ECG.csv',
    'electricity': 'data/electricity.csv',
    'covid': 'data/covid.csv',
    'solar': 'data/solar.csv',
    'metr': 'data/metr.csv',
    'PeMS07':'data/PeMS07.csv',
    'Flight':'data/Flight.csv',
    'weather':'data/weather.csv'
}


# 数据准备
def prepare_data(data_name):
    data_path = DATA_CONFIG[data_name]
    train_set = UnifiedTimeSeriesDataset(
        data_path, 'train', args.seq_len, args.pre_len, scale=True
    )
    scaler = train_set.scaler
    val_set = UnifiedTimeSeriesDataset(
        data_path, 'val', args.seq_len, args.pre_len, scaler=scaler, scale=True
    )
    test_set = UnifiedTimeSeriesDataset(
        data_path, 'test', args.seq_len, args.pre_len, scaler=scaler, scale=True
    )
    return train_set, val_set, test_set
    

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_loss = float('inf')
    patience = 0

    # 创建模型保存目录
    save_dir = os.path.join("output", args.data, "models")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f'best_{args.data}.pth')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        start_time = time.time()

        # 创建带进度条的训练迭代器
        progress_bar = tqdm(enumerate(train_loader), 
                           total=len(train_loader), 
                           desc=f'Epoch {epoch+1}/{epochs}',
                           bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_idx, (x, y) in progress_bar:
            x = x.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.permute(0,2,1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 实时更新进度条信息
            avg_loss = train_loss / (batch_idx + 1)  # 计算平均损失
            progress_bar.set_postfix({
                'batch_loss': f"{loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}"
            })
        
        # 学习率衰减
        if scheduler and (epoch+1) % args.decay_step == 0:
            scheduler.step()

        # 验证
        val_loss, val_mape, val_mae, val_rmse = evaluate(model, val_loader, criterion)
        
        # 早停逻辑
        if args.early_stop:
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                #保存模型
                torch.save(model.state_dict(), best_model_path)
            else:
                patience += 1
                print(f"Val loss not improved for ({patience}/{args.patience}) epochs")
                if patience >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1:3d}")
                    break
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
            
        # 打印日志
        print(f'Epoch {epoch+1:3d} | Time: {time.time()-start_time:5.2f}s | '
              f'Train Loss: {train_loss/len(train_loader):5.4f} | '
              f'Val Loss: {val_loss:5.4f} | '
              f'MAPE: {val_mape:7.4%} | MAE: {val_mae:7.4f} | RMSE: {val_rmse:7.4f}')
        
    print(f"Best model has been saved ")

# 评估函数（返回多指标）
def evaluate(model, loader, criterion):
    model.eval()
    loss_total, preds, trues = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.float().to(device)
            output = model(x)
            loss = criterion(output, y.permute(0,2,1).contiguous())
            loss_total += float(loss)
            preds.append(output.detach().cpu().numpy())
            trues.append(y.permute(0,2,1).contiguous().detach().cpu().numpy())
    
    # 计算指标
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mape, mae, rmse = evaluate_metrics(trues, preds)  
    return loss_total/len(loader), mape, mae, rmse  

# 测试集评估函数（返回多指标 + 可视化）
def evaluate_draw(model, loader, criterion):
    model.eval()
    loss_total, preds, trues = 0, [], []

    inputx = []

    folder_path = './test_results/' + args.data + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.float().to(device), y.float().to(device)
            output = model(x)
            loss = criterion(output, y.permute(0,2,1).contiguous())
            loss_total += float(loss)
            preds.append(output.detach().cpu().numpy())
            trues.append(y.permute(0,2,1).contiguous().detach().cpu().numpy())

            input_current = x.detach().cpu().numpy()
            inputx.append(input_current)
            if i % 200 == 0:
                # 获取当前batch的第一个样本数据
                current_input = input_current[0, :, -1]     # [seq_len]
                current_true = trues[-1][0, :, -1]          # [pre_len]
                current_pred = preds[-1][0, :, -1]          # [pre_len]

                # 拼接输入序列和预测/真实序列
                gt = np.concatenate([current_input, current_true])
                pd = np.concatenate([current_input, current_pred])
                
                # 可视化保存
                visual(gt, pd, os.path.join(folder_path, f'batch_{i}.pdf'))
    
    # 计算指标
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mape, mae, rmse = evaluate_metrics(trues, preds)  
    return loss_total/len(loader), mape, mae, rmse 

#测试集可视化部分
plt.switch_backend('agg')
def visual(true, preds=None, name='./test_results/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.show()
    plt.savefig(name, bbox_inches='tight')


if __name__ == '__main__':

    torch.cuda.empty_cache()        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    train_set, val_set, test_set = prepare_data(args.data)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # 模型初始化
    model = SSMGNN(
        seq_length=args.seq_len,
        pre_length=args.pre_len,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        number_frequency=args.number_frequency,
        feature_size=train_set.data.shape[1]
    ).to(device)
    
    # 优化器与学习率调度
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    
    # 训练
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs)
    
    # 测试
    print("\n=== Training completed. Starting testing ===")
    save_dir = os.path.join("output", args.data, "models")
    best_model_path = os.path.join(save_dir, f'best_{args.data}.pth')

    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Successfully loaded model parameters from {best_model_path} ")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        exit(1)

    # 执行测试
    test_loss, test_mape, test_mae, test_rmse = evaluate_draw(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f} | MAPE: {test_mape:.4%} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}')

    f = open("result_SSMGNN.txt", 'a')
    f.write('seq_len:{}, pre_len:{}\n'.format(args.seq_len, args.pre_len))
    f.write('data:{}, batch_size:{}, embed_size:{}, hidden_size:{}, number_frequency:{}\n'.format(args.data, args.batch_size, args.embed_size, args.hidden_size, args.number_frequency))
    f.write('Test Loss: {:.4f}, MAPE: {:.4%}, MAE: {:.4f}, RMSE: {:.4f} '.format(test_loss, test_mape, test_mae, test_rmse))
    f.write('\n')
    f.write('\n')
    f.close()
    
    torch.cuda.empty_cache()
