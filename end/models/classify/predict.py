import torch
import fitz
from transformers import BertTokenizer, BertForSequenceClassification

def extract_pdf(pdfpath):
    """提取PDF文本内容"""
    text = ""
    try:
        doc = fitz.open(pdfpath)
        for page in doc:
            text += page.get_text()
        
        doc.close()
        text = text.strip()
        return text
    except Exception as e:
        print(f"PDF提取错误: {str(e)}")
        return ""

def split_sentences(text):
    """将文本分割成句子列表"""
    text = text.replace('⚫','').split(" \n")
    #句子列表推导
    sentences = [line.replace('\n','') for line in text if line]
    return sentences

def predict_text(model, tokenizer, text, device, max_length=512):
    """对单个文本进行预测"""
    model.eval()

    # 对文本进行编码
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 将输入移到设备上
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # 获取所有类别的概率
        all_probabilities = probabilities[0].tolist()

    return predicted_class, confidence, all_probabilities


def main():
    """简历分类预测的主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    model_dir = r"C:\Users\xk22l\Desktop\nb\classify\models\best_model"  # 请替换为你的实际路径

    print(f'从 {model_dir} 加载模型...')

    try:
        # 加载模型和分词器
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    sentences = split_sentences(extract_pdf(r"C:/Users/xk22l/Desktop/end/resumes/shuo.pdf"))
    # 类别名称
    class_names = ['基本信息', '教育背景', '工作/项目经历', '个人能力', '其他信息']
    sections = {'基本信息':"",
                '教育背景':"",
                '工作/项目经历':"",
                '个人能力':"",
                '其他信息':""}
    # 对每个句子进行预测
    for text in sentences:
        predicted_class, confidence, all_probabilities = predict_text(
            model, tokenizer, text, device
        )
        if len(text) > 40:
            sections[class_names[predicted_class]] += '\n'+text
        else:
            sections[class_names[predicted_class]] += ';'+text
        # 输出结果
        print(f'\n文本: "{text}"')
        print(f'预测类别: {class_names[predicted_class]}')
        print(f'置信度: {confidence:.4f}')

        print('各类别概率:')
        for i, (class_name, prob) in enumerate(zip(class_names, all_probabilities)):
            print(f'  {class_name}: {prob:.4f}')
    import json
    print(json.dumps(sections, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()