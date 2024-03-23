# 以下是使用 Jupyter Notebook 进行 LLM-DL-KG 实验,利用 Transformers 的 RobertaTokenizer 和 RobertaModel,以及 TorchCRF 的 CRF 模块,在 DuIE 数据集上进行实体识别的完整 Python 代码

```python
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from sklearn.metrics import precision_score, recall_score, f1_score

class DuIEDataset(Dataset):
def **init**(self, file_path, tokenizer, max_length=128):
self.data = self.load_data(file_path)
self.tokenizer = tokenizer
self.max_length = max_length

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['text']
        labels = item['entity']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        offset_mapping = encoding['offset_mapping']

        label_ids = [0] * len(input_ids)
        for label in labels:
            start, end = label['start_index'], label['end_index']
            token_start_index, token_end_index = self.find_token_index(offset_mapping, start, end)
            if token_start_index is not None and token_end_index is not None:
                label_ids[token_start_index] = 1
                label_ids[token_start_index+1:token_end_index+1] = [2] * (token_end_index - token_start_index)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def find_token_index(self, offset_mapping, start, end):
        token_start_index, token_end_index = None, None
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if token_start <= start < token_end:
                token_start_index = i
            if token_start < end <= token_end:
                token_end_index = i
            if token_start_index is not None and token_end_index is not None:
                break
        return token_start_index, token_end_index

class EntityRecognizer(nn.Module):
def **init**(self, pretrained_model, num_labels):
super(EntityRecognizer, self).**init**()
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
for batch in dataloader:
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def evaluate(model, dataloader, device):
model.eval()
true_labels = []
pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            decoded_sequence = model(input_ids, attention_mask)

            true_labels.extend(labels.cpu().numpy().flatten())
            pred_labels.extend(np.array(decoded_sequence).flatten())

    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')

    return precision, recall, f1

# 设置超参数

pretrained_model = 'hfl/chinese-roberta-wwm-ext'
max_length = 128
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

# 加载数据集

tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
train_dataset = DuIEDataset('train.json', tokenizer, max_length)
dev_dataset = DuIEDataset('dev.json', tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

# 初始化模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EntityRecognizer(pretrained_model, num_labels=3).to(device)

# 设置优化器和学习率调度器

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) \* num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型

train(model, train_dataloader, optimizer, scheduler, device, num_epochs)

# 评估模型

precision, recall, f1 = evaluate(model, dev_dataloader, device)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
```

这个示例代码使用了 RoBERTa 作为预训练语言模型,并在顶层添加了 CRF 层来处理实体识别任务。以下是代码的主要组成部分:

DuIEDataset 类:用于加载和处理 DuIE 数据集,将文本转换为模型输入的格式,并生成相应的标签。
EntityRecognizer 类:定义了实体识别模型的结构,使用 RoBERTa 作为编码器,并在顶层添加了一个线性分类器和 CRF 层。
train 函数:定义了模型的训练流程,使用 CRF 的负对数似然损失函数和 AdamW 优化器,并应用了学习率调度器。
evaluate 函数:定义了模型的评估流程,使用 CRF 解码器获取预测的实体序列,并计算精确率、召回率和 F1 值。
主程序部分:
设置超参数,如预训练模型、最大序列长度、批次大小、训练轮数和学习率。
加载 DuIE 数据集,并创建数据加载器。
初始化实体识别模型,并将其移动到相应的设备上。
设置优化器和学习率调度器。
调用 train 函数训练模型。
调用 evaluate 函数评估模型性能。
请注意,这个示例代码假设 DuIE 数据集已经预处理为所需的 JSON 格式,每个样本包含 text 和 entity 字段,其中 entity 字段是一个列表,包含实体的起始和结束位置。你需要根据实际的数据格式进行相应的调整。

在运行代码之前,确保已经安装了所需的库,如 PyTorch、Transformers、TorchCRF 和 scikit-learn。你还需要下载预训练的 RoBERTa 模型,可以通过 Hugging Face 的 Transformers 库获取。

这个示例代码展示了如何使用 RoBERTa 和 CRF 进行实体识别任务,你可以在此基础上进行进一步的优化和改进,如调整超参数、尝试不同的预训练模型、引入其他特征等,以提高实体识别的性能。
