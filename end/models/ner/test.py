import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report

# 加载测试数据
class MSRADataset:
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
                label = -100
            aligned_labels.append(label)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

# 评估函数
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

# 预测函数
def format_entities(predictions):
    """格式化预测结果，合并连续的同类型实体"""
    entities = []
    current_entity = []
    current_type = None
    
    for token, label in predictions:
        if label.startswith('B-'):
            if current_entity:
                entities.append((''.join(current_entity), current_type))
                current_entity = []
            current_entity.append(token)
            current_type = label[2:]
        elif label.startswith('I-') and current_type == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((''.join(current_entity), current_type))
                current_entity = []
            current_type = None
    
    if current_entity:
        entities.append((''.join(current_entity), current_type))
    
    return entities

def predict(text, model, tokenizer, device, id2label):
    """对输入的文本进行NER预测"""
    model.eval()
    tokens = tokenizer.tokenize(text)
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
        preds = torch.argmax(outputs.logits, dim=-1)
    
    # 对齐预测结果
    word_ids = encoding.word_ids()
    predictions = []
    previous_word_idx = None
    for i, (word_idx, pred) in enumerate(zip(word_ids, preds[0])):
        if word_idx is not None and word_idx != previous_word_idx:
            predictions.append((tokens[word_idx], id2label[pred.item()]))
            previous_word_idx = word_idx
    
    # 格式化并合并实体
    return format_entities(predictions)

# 主函数
def main():
    """测试命名实体识别模型的主函数"""
    # 参数设置
    model_path = 'C:/Users/xk22l/Desktop/nb/ner/ner_model'
    # test_data_path = 'msra/test.txt'
    # batch_size = 16
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path).to(device)
    
    # # 加载测试数据
    # test_dataset = MSRADataset(test_data_path, tokenizer)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # # 评估模型
    # report = evaluate(model, test_loader, device, model.config.id2label)
    # print("测试结果:")
    # print(report)
    
    # # 保存测试报告
    # with open(os.path.join(model_path, 'test_report.txt'), 'w') as f:
    #     f.write(report)
    
    # 复杂测试案例
    test_cases = [
        "微软公司位于北京市海淀区，由比尔·盖茨和保罗·艾伦于1975年创立微",
        "中国科学技术大学位于安徽省合肥市，是一所985工程重点大学",
        "苹果公司的CEO蒂姆·库克近日访问了上海迪士尼乐园",
        "北京大学和清华大学都位于北京市海淀区，是中国最著名的高等学府",
        "阿里巴巴集团总部位于浙江省杭州市，由马云在1999年创立",
        "内蒙古自治区呼和浩特市内蒙古大学",
        "姓名：刘永胜\n电话：15044742600\n邮箱：liu_ys597@163.com\n出生年月：2002.07\n政治面貌：共青团员\n籍贯：内蒙古自治区鄂尔多斯市\n业就业情况以及大模型出现对该行业的冲击。\n",
        """教育背景": "本科院校：北京语言大学\n2021.09-2025.06北京语言大学计算机科学与技术（本科）\n主修课程：Python程序设计、C语 言程序设计、面向对象程序设计语言（C++）、数据结构、计算机组成原理、计算机网络、操作系统、算法导论、自然语言处理；\n学习成绩 ：GPA3.26/4，加权平均成绩：83.1;\n报考院校：内蒙古大学报考专业：计算机技术报考属性：专业学位\n数学专业课\n专业课总分\n"""
    ]
    
    for text in test_cases:
        print(f"\n测试文本: {text}")
        predictions = predict(text, model, tokenizer, device, model.config.id2label)
        print("预测结果:")
        for entity, label in predictions:
            print(f"{entity}: {label}")

if __name__ == '__main__':
    main()