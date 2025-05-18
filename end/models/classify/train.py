import json
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',  
    'WenQuanYi Micro Hei', 
    'Microsoft YaHei', 
    'SimHei'
]

class ResumeDataset(Dataset):
    """简历数据集加载和预处理类"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    # 计算混淆矩阵并转换为列表
    cm = confusion_matrix(labels, predictions)
    cm_list = cm.tolist()
    
    # 生成分类报告
    report = classification_report(
        labels, 
        predictions,
        target_names=['基本信息', '教育背景', '工作/项目经历', '个人能力', '其他信息'],
        digits=4,
        output_dict=True
    )
    
    return {
        "accuracy": accuracy,
        "eval_accuracy": accuracy,   # 添加验证准确率
        "f1": f1,
        "confusion_matrix": cm_list,
        "classification_report": report
    }

def plot_training_curves(trainer_state, save_dir):
    """绘制训练曲线"""
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 提取训练和验证数据
    steps = []
    train_losses = []
    train_accs = []
    eval_steps = []
    eval_losses = []
    eval_accs = []
    
    for entry in trainer_state.log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            steps.append(entry['step'])
            train_losses.append(entry['loss'])
            if 'train_accuracy' in entry:
                train_accs.append(entry['train_accuracy'])
        elif 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
            if 'eval_accuracy' in entry:
                eval_accs.append(entry['eval_accuracy'])
    
    # 绘制损失曲线
    ax1.set_title('训练和验证损失')
    # 对训练loss进行移动平均
    window = 5
    if len(train_losses) >= window:
        smoothed_losses = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        ax1.plot(smoothed_steps, smoothed_losses, 'b-', label='训练损失', alpha=0.6, linewidth=1)
    else:
        ax1.plot(steps, train_losses, 'b-', label='训练损失', alpha=0.6, linewidth=1)
    
    if eval_steps:
        # 对验证loss进行插值
        eval_x = np.array(eval_steps)
        eval_y = np.array(eval_losses)
        x_smooth = np.linspace(eval_x.min(), eval_x.max(), 100)
        y_smooth = np.interp(x_smooth, eval_x, eval_y)
        ax1.plot(x_smooth, y_smooth, 'r-', label='验证损失', alpha=0.6, linewidth=1)
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.set_title('验证准确率')
    if eval_accs:
        # 对验证准确率进行插值
        eval_x = np.array(eval_steps)
        eval_y = np.array(eval_accs)
        x_smooth = np.linspace(eval_x.min(), eval_x.max(), 100)
        y_smooth = np.interp(x_smooth, eval_x, eval_y)
        ax2.plot(x_smooth, y_smooth, 'r-', label='验证准确率', alpha=0.6, linewidth=1)
    
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, save_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    # 将列表转换为NumPy数组（如果不是的话）
    if not isinstance(cm, np.ndarray):
        cm = np.array(cm)
    
    labels = ['基本信息', '教育背景', '工作/项目经历', '个人能力', '其他信息']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 调整刻度标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def load_data(file_path):
    """加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    return texts, labels

def main():
    """训练简历分类模型的主函数"""
    # 创建模型保存目录
    output_dir = os.path.join('classify', 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 加载数据
    train_texts, train_labels = load_data('classify/data/train_data.json')
    val_texts, val_labels = load_data('classify/data/val_data.json')
    test_texts, test_labels = load_data('classify/data/test_data.json')
    
    print(f"数据集大小：")
    print(f"训练集：{len(train_texts)}条")
    print(f"验证集：{len(val_texts)}条")
    print(f"测试集：{len(test_texts)}条")
    
    # 初始化tokenizer和模型
    model_path = 'models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f'
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=5,
        local_files_only=True
    )
    
    # 创建数据集
    train_dataset = ResumeDataset(train_texts, train_labels, tokenizer)
    val_dataset = ResumeDataset(val_texts, val_labels, tokenizer)
    test_dataset = ResumeDataset(test_texts, test_labels, tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,   
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=3e-5,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=10,                  
        evaluation_strategy="steps",      
        eval_steps=50,                     
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=4,
        group_by_length=True,
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        report_to=["tensorboard"]
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 训练模型
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(output_dir, 'best_model'))
    
    # 绘制训练曲线
    plot_training_curves(trainer.state, output_dir)
    
    # 评估最终模型
    eval_result = trainer.evaluate(test_dataset)
    
    # 绘制混淆矩阵
    if "eval_confusion_matrix" in eval_result:
        plot_confusion_matrix(eval_result["eval_confusion_matrix"], output_dir)
    
    # 保存评估结果
    with open(os.path.join(output_dir, 'eval_results.txt'), 'w', encoding='utf-8') as f:
        # 首先写入主要指标
        for key in ["eval_loss", "eval_accuracy", "eval_f1"]:
            if key in eval_result:
                value = eval_result[key]
                f.write(f"{key}: {value:.4f}\n")
        
        # 写入详细的分类报告
        f.write("\n=== 详细分类报告 ===\n")
        if "eval_classification_report" in eval_result:
            report = eval_result["eval_classification_report"]
            # 打印每个类别的详细指标
            for label in ['基本信息', '教育背景', '工作/项目经历', '个人能力', '其他信息']:
                if label in report:
                    metrics = report[label]
                    f.write(f"\n{label}:\n")
                    f.write(f"精确率: {metrics['precision']:.4f}\n")
                    f.write(f"召回率: {metrics['recall']:.4f}\n")
                    f.write(f"F1分数: {metrics['f1-score']:.4f}\n")
                    f.write(f"支持度: {metrics['support']}\n")
            
            # 写入整体评估指标
            f.write("\n=== 整体评估 ===\n")
            if 'macro avg' in report:
                f.write(f"宏平均:\n")
                f.write(f"精确率: {report['macro avg']['precision']:.4f}\n")
                f.write(f"召回率: {report['macro avg']['recall']:.4f}\n")
                f.write(f"F1分数: {report['macro avg']['f1-score']:.4f}\n")
            
            if 'weighted avg' in report:
                f.write(f"\n加权平均:\n")
                f.write(f"精确率: {report['weighted avg']['precision']:.4f}\n")
                f.write(f"召回率: {report['weighted avg']['recall']:.4f}\n")
                f.write(f"F1分数: {report['weighted avg']['f1-score']:.4f}\n")

if __name__ == '__main__':
    main()