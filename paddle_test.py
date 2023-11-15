import paddle.nn.functional as F
import paddle.fluid as fluid
from paddle import where
import os
import paddle
# 自定义数据集
import re
from metric import MultiLabelReport

from paddlenlp.datasets import load_dataset

from eval import evaluate
import functools

from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding

# 加载中文ERNIE 3.0预训练模型和分词器
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer


# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length):
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    result["labels"] = examples["labels"]
    return result


def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
    return text


# 定义读取数据集函数
def read_custom_data(filepath='raw_data/test/labeled_0.txt', is_one_hot=True):
    f = open(filepath, encoding="utf-8")
    while True:
        line = f.readline()
        if not line:
            break
        data = line.strip().split('\t')
        # 标签用One-hot表示
        labels = [float(1) if str(i) in data[1].split(',') else float(0) for i in range(20)]
        yield {"text": clean_text(data[0]), "labels": labels}
    f.close()


def load_label_vocab():
    data = open('data/label_mapping.txt', 'r', encoding='utf-8').readlines()
    label_vocab = {}
    for i in data:
        temp = i.split(',')
        label_vocab[int(temp[0])] = temp[1].split('\n')[0]
    return label_vocab


max_seq_len=128
num_classes = 20

path = "ernie_ckpt"
model = AutoModelForSequenceClassification.from_pretrained(path, num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained(path)

trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=max_seq_len)

metric = MultiLabelReport()

collate_fn = DataCollatorWithPadding(tokenizer)

path = 'data/test.txt'
test_ds = load_dataset(read_custom_data, filepath=path, lazy=False)
test_ds = test_ds.map(trans_func)
test_batch_sampler = BatchSampler(test_ds, batch_size=6, shuffle=False)
test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)

auc, eval_f1_score, recall, precision = evaluate(model, metric, test_data_loader, if_return_results=False)
print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, accuracy: %.5f"
      % (precision, recall, eval_f1_score, auc)
      )


label_vocab = load_label_vocab()
results = evaluate(model, metric, test_data_loader, label_vocab)
print(results)
test_ds = load_dataset(read_custom_data, filepath=path, lazy=False)
res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
with open("./results/multi_label.tsv", 'w', encoding="utf-8") as f:
    f.write("text\tprediction\n")
    for i, pred in enumerate(results):
        f.write(test_ds[i]['text']+"\t"+pred+"\n")

1