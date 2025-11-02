# SSMGNN: Spectral Temporal Graph Neural Network with State Space Models for Multivariate Time-series Forecasting

## 🚀 Quick Start

### Requirements

```bash
pip install numpy==1.23.5
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install torch
```

### Train and evaluate SSMGNN
You can use the following command:
```bash
sh train.sh
```

### Dataset
- You can obtain the dataset from： [Baidu Yun](https://pan.baidu.com/s/16bxsa81Hq0Rcdm6W2w1EhQ?pwd=csbg)


## ⚙️ Hyperparameter Configurations
- ECG: -- batch_size 2 -- embed_size 128 -- hidden_size 512 -- feature_blocks 8 -- lr 1e-5 
- Electricity: -- batch_size 2 -- embed_size 256 -- hidden_size 512 -- feature_blocks 4 -- lr 1e-5
- Traffic: -- batch_size 2 -- embed_size 256 -- hidden_size 512 -- feature_blocks 4 -- lr 1e-5
- METR-LA: -- batch_size 32 -- embed_size 128 -- hidden_size 512 -- feature_blocks 8 -- lr 1e-5
- Solar: -- batch_size 2 -- embed_size 256 -- hidden_size 512 -- feature_blocks 64 -- lr 1e-5
- PeMS07: -- batch_size 2 -- embed_size 256 -- hidden_size 512 -- feature_blocks 64 -- lr 1e-5
- COVID-19: -- batch_size 2 -- embed_size 256 -- hidden_size 512 -- feature_blocks 64 -- lr 1e-5
- Flight: -- batch_size 2 -- embed_size 256 -- hidden_size 512 -- feature_blocks 16 -- lr 1e-5
- Weather: -- batch_size 2 -- embed_size 256 -- hidden_size 512 -- feature_blocks 16 -- lr 1e-5

## 🙏 Acknowledgements

We appreciate the following github repos a lot for their valuable code base or datasets:

1. FourierGNN: https://github.com/aikunyi/FourierGNN
2. StemGNN: https://github.com/microsoft/StemGNN
3. MSGNet: https://github.com/YoZhibo/MSGNet
4. CrossGNN: https://github.com/hqh0728/CrossGNN
5. MTGNN: https://github.com/nnzhan/MTGNN
