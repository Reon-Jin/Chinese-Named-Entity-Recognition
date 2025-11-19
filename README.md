## Data
创建一个./data文件夹用于存放数据集

## Usage
训练及测试:
```
python run.py
```
训练时启用Train = True
测试时启用Test = True


## Result
测试结果示例：
```
📊 Overall NER Results:
   Gold:      7201
   Predicted: 6847
   Correct:   5757
   Precision: 0.8408
   Recall:    0.7995
   F1:        0.8196

📌 Per-Entity-Type Results:
Type   P        R        F1       Gold   Pred   Correct
------------------------------------------------------------
ORG    0.7284   0.7515   0.7398   853    880    641
LOC    0.8304   0.7910   0.8102   2761   2630   2184
PER    0.8786   0.8174   0.8469   3587   3337   2932
```

