from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba
import re

# 常用停用词列表
STOP_WORDS = set([
    '的', '了', '和', '是', '就', '都', '而', '及',  '这', '那', '有', '在', '中', '为', '以',
    '上', '下', '个', '等', '对', '能', '可以', '我们', '你们', '他们', '她们', '它们', '自己',
    '什么', '如何', '怎么', '为什么', '哪些', '谁', '哪里', '多少', '几', '很', '太', '又', '也',
    '但', '并', '或', '且', '如果', '因为', '所以', '虽然', '但是', '然而', '因此', '于是',
    '然后', '接着', '首先', '其次', '最后', '总之', '总之', '总之', '总之', '总之', '总之'
])

class Match:
    def __init__(self):
        """初始化语义匹配模型"""
        model_path = r"C:\Users\xk22l\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
        self.model = SentenceTransformer(model_path)

        self.match_config = [
            # 工作经历双匹配
            (['工作/项目经历'], ['岗位职责'], 0.35),
            (['工作/项目经历'], ['任职要求'], 0.15),
            # 个人能力单匹配
            (['个人能力'], ['任职要求'], 0.30),
            # 教育+基本信息联合匹配
            (['教育背景', '基本信息'], ['任职要求'], 0.15),  # 权重降低0.05
            # 其他信息
            (['其他信息'], ['任职要求'], 0.05)
        ]  # 总权重保持1.0

    def _merge_text(self, data):
        """合并文本数据"""
        if isinstance(data, list):
            return ' '.join(str(item) for item in data)
        return str(data)

    def preprocess_resume(self, resume_data):
        """预处理简历数据"""
        processed = {
            '教育背景': '',
            '工作/项目经历': '',
            '个人能力': '',
            '基本信息': '',
            '其他信息': ''
        }
        for block in resume_data.get('blocks', []):
            category = block.get('category', '')
            content = self._merge_text(block.get('content', ''))
            processed[category] = content
        return processed

    def preprocess_job(self, job_data):
        """预处理岗位数据"""
        title = self._merge_text(job_data.get('title', ''))
        desc = f"{title} {self._merge_text(job_data.get('description', ''))}"
        reqs = f"{title} {self._merge_text(job_data.get('requirements', ''))}"

        return {
            '岗位职责': desc.strip(),
            '任职要求': reqs.strip()
        }

    def calculate_similarity(self, resume, job):
        """计算简历和岗位的相似度"""
        total = 0.0
        report = []

        # 提前编码所有字段
        encoded_resume = {k: self.model.encode(v) for k, v in resume.items()}
        encoded_job = {k: self.model.encode(v) for k, v in job.items()}

        for resume_fields, job_fields, weight in self.match_config:
            try:
                # 合并简历向量（加权平均）
                res_vecs = [encoded_resume[rf] for rf in resume_fields]
                res_vec = np.mean(res_vecs, axis=0) if res_vecs else None

                # 合并职位向量（加权平均）
                jd_vecs = [encoded_job[jf] for jf in job_fields]
                jd_vec = np.mean(jd_vecs, axis=0) if jd_vecs else None

                if res_vec is not None and jd_vec is not None:
                    similarity = cosine_similarity([res_vec], [jd_vec])[0][0]
                    contribution = similarity * weight
                    total += contribution

                    report.append({
                        '模块': f"{'+'.join(resume_fields)} vs {'+'.join(job_fields)}",
                        '相似度': round(float(similarity), 2),
                        '贡献值': round(float(contribution), 2)
                    })
            except KeyError:
                continue

        return round(total, 2), report

    def recommend_jobs(self, resume_data, jobs_data, top_k=3):
        """推荐匹配的岗位"""
        processed_resume = self.preprocess_resume(resume_data)
        recommendations = []

        for job in jobs_data:
            processed_job = self.preprocess_job(job)
            score, report = self.calculate_similarity(processed_resume, processed_job)

            recommendations.append({
                '_id': str(job.get('_id')),  # 使用MongoDB的_id字段并转为字符串
                'title': job.get('title'),  # 仍然保留原始title字段用于显示
                'score': float(score),
                'details': report
            })

        # 只保留score>0.4的结果
        filtered = [rec for rec in recommendations if rec['score'] > 0.4]
        filtered.sort(key=lambda x: x['score'], reverse=True)
        if len(filtered) > top_k:
            return filtered[:top_k]
        else:
            return filtered

class JaccardMatch:
    def __init__(self):
        """初始化Jaccard匹配模型"""
        # 初始化jieba分词
        jieba.initialize()

    def _clean_text(self, text):
        """清理文本数据"""
        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        # 分词
        words = jieba.cut(text)
        # 去停用词
        words = [w for w in words if w.strip() and w not in STOP_WORDS]
        return ' '.join(words)

    def _merge_text(self, data):
        """合并文本数据"""
        if isinstance(data, list):
            return ' '.join(str(item) for item in data)
        return str(data)

    def preprocess_resume(self, resume_data):
        """预处理简历数据"""
        all_content = []
        for block in resume_data.get('blocks', []):
            content = self._merge_text(block.get('content', ''))
            all_content.append(content)
        return self._clean_text(' '.join(all_content))

    def preprocess_job(self, job_data):
        """预处理岗位数据"""
        title = self._merge_text(job_data.get('title', ''))
        desc = self._merge_text(job_data.get('description', ''))
        reqs = self._merge_text(job_data.get('requirements', ''))
        return self._clean_text(f"{title} {desc} {reqs}")

    def jaccard_similarity(self, text1, text2):
        """计算两个文本的Jaccard相似度"""
        set1 = set(text1.split())
        set2 = set(text2.split())
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def recommend_jobs(self, resume_data, jobs_data, top_k=3):
        """推荐匹配的岗位"""
        processed_resume = self.preprocess_resume(resume_data)
        recommendations = []
        
        for job in jobs_data:
            processed_job = self.preprocess_job(job)
            similarity = self.jaccard_similarity(processed_resume, processed_job)
            
            recommendations.append({
                '_id': str(job.get('_id')),
                'title': job.get('title'),
                'score': float(similarity),
                'details': [{
                    '模块': '整体内容匹配',
                    '相似度': round(float(similarity), 2),
                    '贡献值': round(float(similarity), 2)
                }]
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]

if __name__ == '__main__':
    # 测试分词以及匹配效果
    def test_tokenization():
        matcher = Match()
        simple_matcher = JaccardMatch()
        
        resume_data = {
            'blocks': [
                {
                    'category': '教育背景',
                    'content': '清华大学        计算机科学与技术硕士（人工智能方向）         2021.09–2024.06专业技能;'
                },
                {
                    'category': '工作/项目经历',
                    'content': '''工作年限：3 年;某头部AI 科技公司            AI 工程师（实习）  2022.07 – 2023.07     
                                参与AI 中台开发，主导3 个企业级RAG 项目落地，客户留存率提升25%
                                推动LangChain 与内部工具链整合，团队开发效率提升30%
                                基于RAG 的智能问答系统（独立开发）   2023.03 – 2024.02      
                                使用LangChain 构建检索流程，集成Milvus 向量数据库，优化Embedding 模型（BAAI/bge），将响应准确率从68%提升至89%
                                设计动态Prompt 模板，结合用户上下文实现多轮对话，推理延迟降低40%（GPU
                                Triton 部署）;多模态文档处理系统（团队核心开发）      2022.06 – 2023.01
                                搭建文档解析后端，集成OCR（PaddleOCR）与NLP 模型（LayoutLM），实现PDF/表格结构化提取（F1=92%）
                                基于LlamaIndex 构建索引，支持语义检索与自动摘要生成，效率提升50%
                                使用LoRA 对Llama2-7B 进行医疗领域微调，通过领域语料增强与奖励模型（PPO）优化，评测指标提升35%
                                部署于华为昇腾910 集群，推理成本降低60%
                                模型优化：设计模型评估框架（准确率/延迟/鲁棒性）、构建监控系统（Prometheus/Grafana）
                                项目经历;工作经历;'''
                },
                {
                    'category': '个人能力',
                    'content': '''AI 开发工具：LangChain、LlamaIndex、OpenAI/Claude API，熟悉Prompt Engi-neering 最佳实践
                                算法与框架：PyTorch/TensorFlow，精通RAG、微调（LoRA）、向量数据库（Milvus/Pinecone）
                                工程能力：AI Agent 系统设计、Docker/Kubernetes 部署、AWS/Azure 云服务、国产化环境适配（华为昇腾）
                                语言：英语CET-6（流利技术文档读写），熟练使用Markdown/Sphinx 编写技术文档'''
                },
                {
                    'category': '基本信息',
                    'content': '鲍凡;意向岗位：AI 工程师;出生日期：1960.05 ;籍贯：湖北省襄樊市;电话：15201401579;邮箱：3dpme@live.com ;技术方案入选公司国产化替代白皮书;'
                },
                {
                    'category': '其他信息',
                    'content': '兴趣爱好;编程、看电影、音乐;•;•;1.;2.;1.;2.;3.;大模型垂直领域微调（主导）                     2023.09 – 2023.121.;2.;•;•;•;•;•;教育背景;'
                }
            ]
        }
        
        job_data = {
            'title': 'ai工程师',
            'description': '''1. 深度参与Ai平台的架构设计和开发实现；
                2. 研究主流人工智能产品的应用实现，完成产品开发落地及迭代优化，参与系统性能的优化；
                3. 设计人工智能产品在国产化计算环境下的技术方案；
                4. 通过提示词工程、RAG和微调等手段优化模型输出质量；
                5. 集成和部署AI模型API，确保高性能和可扩展性；
                6. 开发和维护模型评估框架和监控系统；
                7. 与产品和业务团队合作，将AI功能整合到现有产品中；
                8. 编写技术文档和最佳实践指南；
                9. 跟踪最新的AI技术发展动态，探索和跟进前沿技术/算法，推动应用落地；''',
            'requirements': '''1.熟练使用主流 AI 开发具，如 LangChain 、 Llamalndex 、 OpenAI API 、 Claude API 等，了解 Prompt Engineering 最佳实践和技巧；
                2.具备 Al Agent 系统设计经验，能够构建基于 RAG （检索增强生成）的解决方案，掌握向量数据库的使用方法；
                3.具备以下经验者优先：实时协作系统或文档处理后端的开发经验；成熟的 AI 应用开发经验； GitHub 贡献记录或维护个人技术项目的经验等；
                4.强烈的 owner 意识，具备优秀的跨团队协同能力，主导过5人以上研发团队协作经验优先。
                5.985/211硕士学历优先'''
        }
        
        # 处理简历数据
        print("【简历数据处理效果】")
        processed_resume = simple_matcher.preprocess_resume(resume_data)
        print("\n简历整体内容分词后：")
        print(processed_resume)
        
        # 处理职位数据
        print("\n【职位数据处理效果】")
        processed_job = simple_matcher.preprocess_job(job_data)
        print("\n职位整体内容分词后：")
        print(processed_job)
        
        # 计算相似度
        similarity = simple_matcher.jaccard_similarity(processed_resume, processed_job)
        print(f"\n整体相似度：{similarity:.4f}")
        
        # 输出两种匹配方式的结果
        print("\n" + "="*50 + "\n")
        print('【语义余弦匹配】')
        print(matcher.recommend_jobs(resume_data, [job_data]))
        print('\n【简单Jaccard匹配】')
        print(simple_matcher.recommend_jobs(resume_data, [job_data]))
    
    test_tokenization()