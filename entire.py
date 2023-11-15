import os
import paddle
# 自定义数据集
import re

from paddlenlp.datasets import load_dataset

from eval import evaluate
import functools

from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding

# 加载中文ERNIE 3.0预训练模型和分词器
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

import time
import paddle.nn.functional as F

from metric import MultiLabelReport


def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
    return text

# 定义读取数据集函数
def read_custom_data(is_test=False, is_one_hot=True):

    file_num = 1# 6 if is_test else 48
    filepath = 'raw_data/test/' if is_test else 'raw_data/train/'

    for i in range(file_num):
        f = open('{}labeled_{}.txt'.format(filepath, i), encoding="utf-8")
        while True:
            line = f.readline()
            if not line:
                break
            data = line.strip().split('\t')
            # 标签用One-hot表示
            if is_one_hot:
                labels = [float(1) if str(i) in data[1].split(',') else float(0) for i in range(20)]
            else:
                labels = [int(d) for d in data[1].split(',')]
            yield {"text": clean_text(data[0]), "labels": labels}
        f.close()

label_vocab = {
    0: "婚后有子女",
    1: "限制行为能力子女抚养",
    2: "有夫妻共同财产",
    3: "支付抚养费",
    4: "不动产分割",
    5: "婚后分居",
    6: "二次起诉离婚",
    7: "按月给付抚养费",
    8: "准予离婚",
    9: "有夫妻共同债务",
    10: "婚前个人财产",
    11: "法定离婚",
    12: "不履行家庭义务",
    13: "存在非婚生子",
    14: "适当帮助",
    15: "不履行离婚协议",
    16: "损害赔偿",
    17: "感情不和分居满二年",
    18: "子女随非抚养权人生活",
    19: "婚后个人财产"
}
# load_dataset()创建数据集
train_ds = load_dataset(read_custom_data, is_test=False, lazy=False)
test_ds = load_dataset(read_custom_data, is_test=True, lazy=False)

# lazy=False，数据集返回为MapDataset类型
print("数据类型:", type(train_ds))

# labels为One-hot标签
print("训练集样例:", train_ds[0])
print("测试集样例:", test_ds[0])



model_name = "ernie-3.0-base-zh"
num_classes = 20
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length):
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    result["labels"] = examples["labels"]
    return result


trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=128)
train_ds = train_ds.map(trans_func)
test_ds = test_ds.map(trans_func)

# collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
collate_fn = DataCollatorWithPadding(tokenizer)

# 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
train_batch_sampler = BatchSampler(train_ds, batch_size=2, shuffle=True)
test_batch_sampler = BatchSampler(test_ds, batch_size=2, shuffle=False)
train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)

# Adam优化器、交叉熵损失函数、自定义MultiLabelReport评价指标
optimizer = paddle.optimizer.AdamW(learning_rate=1e-4, parameters=model.parameters())
criterion = paddle.nn.BCEWithLogitsLoss()
metric = MultiLabelReport()

epochs = 10  # 训练轮次
ckpt_dir = "ernie_ckpt"  # 训练过程中保存模型参数的文件夹

global_step = 0  # 迭代次数
tic_train = time.time()
best_f1_score = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

        # 计算模型输出、损失函数值、分类概率值、准确率、f1分数
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.sigmoid(logits)
        metric.update(probs, labels)
        auc, f1_score, _, _ = metric.accumulate()

        # 每迭代10次，打印损失函数值、准确率、f1分数、计算速度
        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, auc: %.5f, f1 score: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, auc, f1_score,
                   10 / (time.time() - tic_train)))
            tic_train = time.time()

        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # 每迭代40次，评估当前训练的模型、保存当前最佳模型参数和分词器的词表等
        if global_step % 40 == 0:
            save_dir = ckpt_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            eval_f1_score = evaluate(model, criterion, metric, test_data_loader, label_vocab, if_return_results=False)
            if eval_f1_score > best_f1_score:
                best_f1_score = eval_f1_score
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)


# 模型在测试集中表现
model.set_dict(paddle.load('ernie_ckpt/model_state.pdparams'))
print("ERNIE 3.0 在法律文本多标签分类test集表现", end= " ")
results = evaluate(model, criterion, metric, test_data_loader, label_vocab)
print(results)
test_ds = load_dataset(read_custom_data, is_test=True, is_one_hot=False, lazy=False)
res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
with open(os.path.join(res_dir, "multi_label.tsv"), 'w', encoding="utf8") as f:
    f.write("text\tprediction\n")
    for i, pred in enumerate(results):
        f.write(test_ds[i]['text']+"\t"+pred+"\n")

