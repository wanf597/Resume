import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report
from tqdm import tqdm

# 1. 数据预处理
class MSRADataset(Dataset):
    """MSRA数据集加载和预处理类"""
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        self.label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def load_data(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            labels = []
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        data.append((tokens, labels))
                        tokens = []
                        labels = []
                    continue
                token, label = line.split()
                tokens.append(token)
                labels.append(label)
            if tokens:
                data.append((tokens, labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 对齐标签
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label = -100
            elif word_idx != previous_word_idx:
                label = self.label2id[labels[word_idx]]
            else:
                label = -100  # 只标记每个单词的第一个token
            aligned_labels.append(label)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

# 2. 训练函数
def train(model, dataloader, optimizer, device):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 3. 评估函数
def evaluate(model, dataloader, device, id2label):
    """评估模型性能"""
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            # 收集预测和真实标签
            for i in range(len(preds)):
                pred = [id2label[p.item()] for p, l in zip(preds[i], labels[i]) if l != -100]
                true = [id2label[l.item()] for l in labels[i] if l != -100]
                predictions.append(pred)
                true_labels.append(true)
    
    return classification_report(true_labels, predictions)

# 4. 主函数
def main():
    """训练命名实体识别模型的主函数"""
    # 参数设置
    model_name = 'bert-base-chinese'
    data_path = 'msra'  # 数据路径
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5
    output_dir = './ner_model'
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=7,
        id2label={0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'},
        label2id={'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
    ).to(device)
    
    # 数据加载
    train_dataset = MSRADataset(os.path.join(data_path, 'train.txt'), tokenizer)
    dev_dataset = MSRADataset(os.path.join(data_path, 'test.txt'), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    train_losses = []
    eval_reports = []
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        
        # 训练阶段
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估阶段
        report = evaluate(model, dev_loader, device, model.config.id2label)
        eval_reports.append(report)
        
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(report)
        
        # 保存每个epoch的指标
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'training_log.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}/{epochs}\n')
            f.write(f'Train Loss: {avg_train_loss:.4f}\n')
            f.write(f'{report}\n\n')
    
    # 保存模型
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()