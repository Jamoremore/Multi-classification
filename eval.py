import paddle.nn.functional as F
import paddle.fluid as fluid
from paddle import where


def evaluate(model, metric, test_data_loader, label_vocab={}, if_return_results=True):
    if if_return_results:
        result = []
        for step, batch in enumerate(test_data_loader, start=1):
            input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

            # 计算模型输出、损失函数值、分类概率值、准确率、f1分数
            logits = model(input_ids, token_type_ids)
            probs = F.sigmoid(logits)
            metric.update(probs, labels)
            # 减去0.5，得到一个新Tensor，其中原Tensor对应元素大于0.5的位置现在为正数，小于0.5的为负数
            for i in probs:
                temp = ''
                for j in range(len(i)):
                    if i[j] > 0.5:
                        temp = temp+label_vocab[j]+','
                if len(temp)>1:
                    temp = temp[:len(temp)-1]+'\n'
                else:
                    temp = temp + '\n'
                result.append(temp)
        auc, f1_score, recall, precision = metric.accumulate()
        print(f1_score)
        return result
    else:
        for step, batch in enumerate(test_data_loader, start=1):
            input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

            # 计算模型输出、损失函数值、分类概率值、准确率、f1分数
            logits = model(input_ids, token_type_ids)
            # loss = criterion(logits, labels)
            probs = F.sigmoid(logits)
            metric.update(probs, labels)
        auc, f1, recall, precision = metric.accumulate()
        return auc, f1, recall, precision
