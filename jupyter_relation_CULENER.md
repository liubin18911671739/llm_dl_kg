# 以下是在 Jupyter Notebook 中利用 Transformers 的 RobertaTokenizer 和 RobertaModel 模块实现 LLM-DL-KG 方法,对 CLUENER 数据集进行关系抽取的完整 Python 代码

```python
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

class CLUENERDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = example['text']
        relations = example['relations']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'relations': relations
        }

class KnowledgeRepresentation(nn.Module):
    def __init__(self, pretrained_model):
        super(KnowledgeRepresentation, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

class RelationExtractor(nn.Module):
    def __init__(self, pretrained_model, num_relations):
        super(RelationExtractor, self).__init__()
        self.knowledge_rep = KnowledgeRepresentation(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.knowledge_rep.roberta.config.hidden_size, num_relations)

    def forward(self, input_ids, attention_mask):
        knowledge_rep = self.knowledge_rep(input_ids, attention_mask)
        pooled_output = knowledge_rep[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train(model, dataloader, optimizer, scheduler, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            relations = batch['relations']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            labels = torch.zeros(logits.size(), dtype=torch.float32).to(device)
            for i, rel_list in enumerate(relations):
                for rel in rel_list:
                    labels[i, rel] = 1

            loss = nn.BCEWithLogitsLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            relations = batch['relations']

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            true_labels.extend([rel for rel_list in relations for rel in rel_list])
            pred_labels.extend([i for i, p in enumerate(probs.cpu().numpy().tolist()) if p >= threshold])

    precision = precision_score(true_labels, pred_labels, average='micro')
    recall = recall_score(true_labels, pred_labels, average='micro')
    f1 = f1_score(true_labels, pred_labels, average='micro')

    return precision, recall, f1

# 设置随机种子
random.seed(42)
torch.manual_seed(42)

# 设置超参数
pretrained_model = 'hfl/chinese-roberta-wwm-ext'
max_length = 128
batch_size = 16
num_epochs = 5
learning_rate = 2e-5
threshold = 0.5

# 加载数据集
train_file = 'path/to/cluener/train.json'
dev_file = 'path/to/cluener/dev.json'
relation_types = ['Cause-Effect', 'Component-Whole', 'Content-Container', 'Entity-Destination', 'Entity-Origin', 'Instrument-Agency', 'Member-Collection', 'Message-Topic', 'Product-Producer']

tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
train_dataset = CLUENERDataset(train_file, tokenizer, max_length)
dev_dataset = CLUENERDataset(dev_file, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RelationExtractor(pretrained_model, num_relations=len(relation_types)).to(device)

# 设置优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
train(model, train_dataloader, optimizer, scheduler, device, num_epochs)

# 评估模型
precision, recall, f1 = evaluate(model, dev_dataloader, device, threshold)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
```

这个示例代码展示了如何使用 LLM-DL-KG 方法在 CLUENER 数据集上进行关系抽取任务。以下是代码的主要组成部分:

1. `CLUENERDataset` 类:用于加载和处理 CLUENER 数据集,将文本转换为模型输入的格式,并返回关系标签。

2. `KnowledgeRepresentation` 类:使用 RoBERTa 模型获取输入文本的知识表示。

3. `RelationExtractor` 类:关系抽取模型,包括知识表示模块和分类器。

4. `train` 函数:定义了模型的训练流程,使用 BCE 损失函数和 AdamW 优化器。

5. `evaluate` 函数:定义了模型的评估流程,计算精确率、召回率和 F1 值。

6. 主程序部分:
   - 设置随机种子,确保结果可重复。
   - 设置超参数,如预训练模型、最大序列长度、批次大小、训练轮数、学习率和阈值。
   - 加载 CLUENER 数据集,并创建数据加载器。
   - 初始化关系抽取模型,并将其移动到相应的设备上。
   - 设置优化器和学习率调度器。
   - 调用 `train` 函数训练模型。
   - 调用 `evaluate` 函数评估模型性能。

请注意,这个示例代码假设 CLUENER 数据集已经进行了预处理,并且每个样本都包含 `text` 和 `relations` 字段。`relations` 字段是一个列表,其中包含文本中存在的关系类型。你需要根据实际的数据格式进行相应的调整。

在运行代码之前,确保已经安装了所需的库,如 PyTorch、Transformers、scikit-learn 等。此外,将数据集文件的路径替换为你自己的数据集路径。

这个示例代码展示了如何使用 LLM-DL-KG 方法在 CLUENER 数据集上进行关系抽取任务。你可以根据需要对代码进行修改和优化,如调整超参数、尝试不同的预训练模型、引入其他特征等,以提高关系抽取的性能。
