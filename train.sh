#!/bin/sh

# SSMGNN 多数据集训练脚本
# 运行此脚本将执行 9个数据集 × 3个随机种子 = 27个实验

# 设置日志目录
LOG_DIR="training_logs"
mkdir -p $LOG_DIR

# 定义数据集参数（使用变量而不是关联数组）
ECG_PARAMS="2 128 512 8"
ELECTRICITY_PARAMS="2 256 512 4"
TRAFFIC_PARAMS="2 256 512 4"
METR_PARAMS="32 128 512 8"
SOLAR_PARAMS="2 256 512 64"
PEMS07_PARAMS="2 256 512 64"
COVID_PARAMS="2 256 512 64"
FLIGHT_PARAMS="2 256 512 16"
WEATHER_PARAMS="2 256 512 64"

# 数据集列表
DATASETS="ECG electricity traffic metr solar PeMS07 covid Flight weather"



# 公共参数
SEQ_LEN=12
PRE_LEN=12
EPOCHS=100
LR=1e-5
DECAY_STEP=5
DECAY_RATE=0.5

echo "开始 SSMGNN 多数据集训练..."
# echo "总共运行: 9个数据集 × 3个随机种子 = 27 个实验"
echo "=========================================="

# 遍历所有数据集
for dataset in $DATASETS; do
    # 获取对应数据集的参数
    case $dataset in
        "ECG") params=$ECG_PARAMS ;;
        "electricity") params=$ELECTRICITY_PARAMS ;;
        "traffic") params=$TRAFFIC_PARAMS ;;
        "metr") params=$METR_PARAMS ;;
        "solar") params=$SOLAR_PARAMS ;;
        "PeMS07") params=$PEMS07_PARAMS ;;
        "covid") params=$COVID_PARAMS ;;
        "Flight") params=$FLIGHT_PARAMS ;;
        "weather") params=$WEATHER_PARAMS ;;
    esac
    
    # 解析参数
    BATCH_SIZE=$(echo $params | cut -d' ' -f1)
    EMBED_SIZE=$(echo $params | cut -d' ' -f2)
    HIDDEN_SIZE=$(echo $params | cut -d' ' -f3)
    NUMBER_FREQUENCY=$(echo $params | cut -d' ' -f4)
    
    echo "正在处理数据集: $dataset"
    echo "参数: batch_size=$BATCH_SIZE, embed_size=$EMBED_SIZE, hidden_size=$HIDDEN_SIZE, number_frequency=$NUMBER_FREQUENCY"
    
    # 遍历所有随机种子
    # for seed in 42 10 100; do
    for seed in 42; do
        echo "--- 运行实验: $dataset (seed=$seed) ---"
        
        # 设置日志文件
        LOG_FILE="$LOG_DIR/${dataset}_seed${seed}.log"
        
        # 执行训练命令
        python main.py \
            --data "$dataset" \
            --seq_len $SEQ_LEN \
            --pre_len $PRE_LEN \
            --batch_size $BATCH_SIZE \
            --embed_size $EMBED_SIZE \
            --hidden_size $HIDDEN_SIZE \
            --number_frequency $NUMBER_FREQUENCY \
            --epochs $EPOCHS \
            --lr $LR \
            --decay_step $DECAY_STEP \
            --decay_rate $DECAY_RATE \
            --early_stop \
            --patience 7 \
            --seed $seed \
            > $LOG_FILE 2>&1
        
        # 检查训练是否成功完成
        if [ $? -eq 0 ]; then
            echo "✓ $dataset (seed=$seed) 训练完成"
        else
            echo "✗ $dataset (seed=$seed) 训练失败"
        fi
        
        echo "日志保存至: $LOG_FILE"
        echo "------------------------------------------"
        
        # 可选：添加延迟以避免资源冲突
        sleep 10
    done
    
    echo "数据集 $dataset 的所有实验完成"
    echo "=========================================="
done

echo "所有实验运行完成！"
echo "实验结果保存在 $LOG_DIR 目录中"

# 生成实验总结
echo "生成实验总结..."
SUMMARY_FILE="$LOG_DIR/experiment_summary.txt"
echo "SSMGNN 多数据集实验总结" > $SUMMARY_FILE
echo "运行时间: $(date)" >> $SUMMARY_FILE
echo "==========================================" >> $SUMMARY_FILE

for dataset in $DATASETS; do
    case $dataset in
        "ECG") params=$ECG_PARAMS ;;
        "electricity") params=$ELECTRICITY_PARAMS ;;
        "traffic") params=$TRAFFIC_PARAMS ;;
        "metr") params=$METR_PARAMS ;;
        "solar") params=$SOLAR_PARAMS ;;
        "PeMS07") params=$PEMS07_PARAMS ;;
        "covid") params=$COVID_PARAMS ;;
        "Flight") params=$FLIGHT_PARAMS ;;
        "weather") params=$WEATHER_PARAMS ;;
    esac
    
    BATCH_SIZE=$(echo $params | cut -d' ' -f1)
    EMBED_SIZE=$(echo $params | cut -d' ' -f2)
    HIDDEN_SIZE=$(echo $params | cut -d' ' -f3)
    NUMBER_FREQUENCY=$(echo $params | cut -d' ' -f4)
    
    echo "数据集: $dataset" >> $SUMMARY_FILE
    echo "参数: batch_size=$BATCH_SIZE, embed_size=$EMBED_SIZE, hidden_size=$HIDDEN_SIZE, number_frequency=$NUMBER_FREQUENCY" >> $SUMMARY_FILE
    # echo "随机种子: 42 10 100" >> $SUMMARY_FILE
    echo "随机种子: 42" >> $SUMMARY_FILE
    echo "---" >> $SUMMARY_FILE
done

echo "实验总结已保存至: $SUMMARY_FILE"