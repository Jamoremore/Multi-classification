# 文本多分类模型

参考基于paddlenlp的ernie3.0、基于pytorch的bert的单分类模型项目做了整合，实现了美妆评论数据的多分类模型训练、预测、测试。

## 参考项目

https://github.com/PaddlePaddle/PaddleNLP

## 文件结构

| 文件名                 | 功能     |
| :--------------------- | -------- |
| **train.py**           | 模型训练 |
| **data_process.py**    | 生成数据 |
| **inference.py**       | 快速测试 |
| **iTrainingLogger.py** | 绘图     |

## 运行
```
python train.py
```
