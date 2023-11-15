from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from paddle import reshape,zeros,concat,tolist


class MultiLabelReport:
    def __init__(self):
        self.probs = []
        self.labels = []

    def update(self, probs, labels):
        self.probs.extend((probs > 0.5).tolist()[0])
        self.labels.extend(labels.tolist()[0])

    def accumulate(self, round_num=3):
        pre = self.probs
        lab = self.labels
        auc = round(accuracy_score(pre, lab), round_num)
        f1 = round(f1_score(pre, lab, average='weighted'), round_num)
        recall = round(recall_score(pre, lab, average='weighted'), round_num)
        precision = round(precision_score(pre, lab, average='weighted'), round_num)
        return auc, f1, recall, precision
