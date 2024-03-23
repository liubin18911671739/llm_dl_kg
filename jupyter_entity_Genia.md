# 以下是使用 Jupyter Notebook 进行 LLM-DL-KG 方法测试,利用 Transformers 的 RobertaTokenizer 和 RobertaModel,以及 TorchCRF 的 CRF 模块,在 Genia 数据集上进行实体识别的完整 Python 代码

```python
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

class GeniaDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=128, label_map=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map
        self.examples = self.load_data()

    def load_data(self):
        examples = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(self.data_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    tokens = text.split()
                    labels = ['O'] * len(tokens)

                    annotation_file = file[:-4] + '.a1'
                    annotation_path = os.path.join(self.data_dir, annotation_file)
                    if os.path.exists(annotation_path):
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith('T'):
                                    fields = line.split()
                                    entity_type = fields[1]
                                    start, end = map(int, fields[2].split())
                                    labels[start] = f'B-{entity_type}'
                                    for i in range(start + 1, end):
                                        labels[i] = f'I-{entity_type}'

                    examples.append((tokens, labels))

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        tokens, labels = self.examples[index]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        label_ids = [self.label_map[label] for label in labels]
        label_ids = [self.label_map['O']] + label_ids + [self.label_map['O']]
        label_ids = label_ids[:self.max_length]
        label_ids = label_ids + [self.label_map['O']] * (self.max_length - len(label_ids))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class EntityRecognizer(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(EntityRecognizer, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(logits, labels)
            return loss
        else:
            decoded_sequence = self.crf.decode(logits)
            return decoded_sequence

def train(model, dataloader, optimizer, scheduler, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

def evaluate(model, dataloader, device, label_map):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            decoded_sequence = model(input_ids, attention_mask)

            true_labels.extend(labels.cpu().numpy().flatten())
            pred_labels.extend([label for seq in decoded_sequence for label in seq])

    true_labels = [label for label in true_labels if label != label_map['O']]
    pred_labels = [label for label in pred_labels if label != label_map['O']]

    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')

    return precision, recall, f1

# 设置随机种子
random.seed(42)
torch.manual_seed(42)

# 设置超参数
pretrained_model = 'roberta-base'
max_length = 128
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

# 加载数据集
data_dir = 'path/to/genia/dataset'
label_list = ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-protein', 'I-protein', 'B-cell_line', 'I-cell_line', 'B-cell_type', 'I-cell_type']
label_map = {label: i for i, label in enumerate(label_list)}

tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
train_dataset = GeniaDataset(data_dir + '/train', tokenizer, max_length, label_map)
dev_dataset = GeniaDataset(data_dir + '/dev', tokenizer, max_length, label_map)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EntityRecognizer(pretrained_model, num_labels=len(label_list)).to(device)

# 设置优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
train(model, train_dataloader, optimizer, scheduler, device, num_epochs)

# 评估模型
precision, recall, f1 = evaluate(model, dev_dataloader, device, label_map)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
```

这个示例代码使用了 RoBERTa 作为预训练语言模型,并在顶层添加了 CRF 层来处理实体识别任务。以下是代码的主要组成部分:

1. `GeniaDataset` 类:用于加载和处理 Genia 数据集,将文本转换为模型输入的格式,并生成相应的标签。

2. `EntityRecognizer` 类:定义了实体识别模型的结构,使用 RoBERTa 作为编码器,并在顶层添加了一个线性分类器和 CRF 层。

3. `train` 函数:定义了模型的训练流程,使用 CRF 的负对数似然损失函数和 AdamW 优化器,并应用了学习率调度器。

4. `evaluate` 函数:定义了模型的评估流程,使用 CRF 解码器获取预测的实体序列,并计算精确率、召回率和 F1 值。

5. 主程序部分:
   - 设置随机种子,确保结果可重复。
   - 设置超参数,如预训练模型、最大序列长度、批次大小、训练轮数和学习率。
   - 加载 Genia 数据集,并创建数据加载器。
   - 初始化实体识别模型,并将其移动到相应的设备上。
   - 设置优化器和学习率调度器。
   - 调用 `train` 函数训练模型。
   - 调用 `evaluate` 函数评估模型性能。

请注意,这个示例代码假设 Genia 数据集已经按照特定的格式组织,每个示例包含一个文本文件(`.txt`)和相应的标注文件(`.a1`)。你需要根据实际的数据格式进行相应的调整。

在运行代码之前,确保已经安装了所需的库,如 PyTorch、Transformers、TorchCRF 和 scikit-learn。你还需要下载预训练的 RoBERTa 模型,可以通过 Hugging Face 的 Transformers 库获取。

这个示例代码展示了如何使用 RoBERTa 和 CRF 在 Genia 数据集上进行实体识别任务。你可以根据需要对代码进行修改和优化,如调整超参数、尝试不同的预训练模型、引入其他特征等,以提高实体识别的性能。
