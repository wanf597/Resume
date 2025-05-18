from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
import os
from sectionDivide import (
    extract_pdf, allowed_file,
    extract_docx,rule_parser,bert_parser
)
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import json
from match import Match,JaccardMatch
import numpy as np
from database import MongoDB
from bson.objectid import ObjectId

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """处理numpy数据类型的JSON序列化"""
        if isinstance(obj, np.integer):  # 处理整数类型（如np.int64）
            return int(obj)
        elif isinstance(obj, np.floating):  # 处理浮点类型（如np.float64）
            return float(obj)
        elif isinstance(obj, np.ndarray):  # 处理数组
            return obj.tolist()
        else:
            return super().default(obj)  # 其他类型按默认处理

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        """处理MongoDB ObjectId的JSON序列化"""
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(MongoJSONEncoder, self).default(obj)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = '8f42a73054b1749f8f58848be5e6502c8e6f3c2d'  # 用于session加密
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # 设置session 7天后过期
app.json_encoder = MongoJSONEncoder  # 设置自定义的JSON编码器

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化MongoDB连接
db = MongoDB()

@app.route('/')
def index():
    """首页路由"""
    if 'user_id' in session:
        user_type = session.get('user_type')
        if user_type == 'jobseeker':
            return redirect(url_for('jobseeker_home'))
        elif user_type == 'recruiter':
            return redirect(url_for('recruiter_home'))
        elif user_type == 'admin':
            return redirect(url_for('admin'))
    return render_template('login.html')

@app.route('/register')
def register():
    """注册页面路由"""
    return render_template('register.html')

@app.route('/jobseeker_home')
def jobseeker_home():
    """求职者首页路由"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return redirect(url_for('index'))
    return render_template('jobseeker_home.html')

@app.route('/recruiter_home')
def recruiter_home():
    """招聘者首页路由"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return redirect(url_for('index'))
    return render_template('recruiter_home.html')

@app.route('/api/login', methods=['POST'])
def login():
    """用户登录API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
            
        user_type = data.get('userType')
        username = data.get('username')  # 可能是用户名或邮箱
        password = data.get('password')
        remember_me = data.get('rememberMe', False)
        
        print(f"登录尝试: userType={user_type}, username={username}")
        
        if not all([username, password]):
            return jsonify({'success': False, 'error': '请填写所有必填字段'}), 400
        
        # 先尝试管理员登录
        if username == 'admin':
            admin_user = db.users.find_one({
                'username': username,
                'password': password,
                'userType': 'admin'
            })
            if admin_user:
                session['user_id'] = str(admin_user['_id'])
                session['user_type'] = 'admin'
                session['username'] = admin_user['username']
                if remember_me:
                    session.permanent = True
                print(f"管理员登录成功: {username}")
                return jsonify({'success': True, 'redirect': url_for('admin')})
        
        # 如果不是管理员，则验证普通用户
        if not user_type:
            return jsonify({'success': False, 'error': '请选择用户类型'}), 400
            
        # 从MongoDB中查找用户（支持用户名或邮箱登录）
        user = db.users.find_one({
            '$and': [
                {'userType': user_type},
                {'password': password},
                {'$or': [
                    {'username': username},
                    {'email': username}
                ]}
            ]
        })
        
        if user:
            if user.get('status', 'active') != 'active':
                return jsonify({'success': False, 'error': '账号已被禁用，请联系管理员'}), 403
            session['user_id'] = str(user['_id'])
            session['user_type'] = user['userType']
            session['username'] = user['username']
            if remember_me:
                session.permanent = True
            print(f"登录成功: {username}")
            if user_type == 'jobseeker':
                return jsonify({'success': True, 'redirect': url_for('jobseeker_home')})
            else:
                return jsonify({'success': True, 'redirect': url_for('recruiter_home')})
        
        print(f"登录失败: 用户名/邮箱或密码错误")
        return jsonify({'success': False, 'error': '用户名/邮箱或密码错误'}), 401
        
    except Exception as e:
        print(f"登录过程发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout')
def logout():
    """用户登出API"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/jobseeker/recommend', methods=['GET'])
def jobseeker_recommend():
    """求职者推荐页面路由"""
    if 'user_id' not in session or session.get('user_type') not in ['jobseeker', 'admin']:
        return redirect(url_for('index'))
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    now = datetime.now()
    # 管理员不做订阅限制
    if session.get('user_type') == 'admin':
        return render_template('job_recommend.html')
    if not user or not user.get('is_subscribed', False) or not user.get('subscribe_expire') or now > user.get('subscribe_expire'):
        return redirect(url_for('subscribe'))
    return render_template('job_recommend.html')


@app.route('/api/recruiter/search', methods=['GET'])
def talents_search():
    """人才搜索页面路由"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return redirect(url_for('index'))
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    now = datetime.now()
    if not user or not user.get('is_subscribed', False) or not user.get('subscribe_expire') or now > user.get('subscribe_expire'):
        return redirect(url_for('subscribe'))
    return render_template('talents_search.html')

@app.route('/api/recruiter/save_job', methods=['POST'])
def save_job():
    """保存招聘岗位信息"""
    try:
        # 检查用户是否登录
        if 'user_id' not in session or session.get('user_type') != 'recruiter':
            return jsonify({'success': False, 'error': '请先登录'})
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'})
            
        # 验证必要字段
        required_fields = ['title', 'description', 'requirements', 'salary_range', 'location', 'company']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'缺少必要字段: {field}'})
                
        # 添加元数据
        job_data = {
            'title': data['title'],
            'description': data['description'],
            'requirements': data['requirements'],
            'salary_range': data.get('salary_range', ''),
            'location': data.get('location', ''),
            'company': data.get('company', ''),
            'created_at': datetime.now(),
            'recruiter_id': ObjectId(session['user_id']),  # 转换为ObjectId
            'status': 'inactive',  # 初始状态为未激活
            'audit_status': 'pending',  # 初始审核状态为待审核
            'audit_comment': None,  # 审核意见
            'audit_time': None  # 审核时间
        }
        
        # 保存到MongoDB
        result = db.jobs.insert_one(job_data)
            
        return jsonify({
            'success': True,
            'message': '岗位已提交审核',
            'data': {
                'job_id': str(result.inserted_id)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/jobseeker/upload', methods=['POST'])
def upload_resume():
    """上传简历API"""
    try:
        if 'user_id' not in session or session.get('user_type') not in ['jobseeker', 'admin']:
            return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请选择上传文件'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return jsonify({
                'success': True,
                'filename': filename
            })
        return jsonify({'success': False, 'error': '不支持的文件类型'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/jobseeker/parse', methods=['POST'])
def parse_resume():
    """解析简历API"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        print(f"\n=== 开始解析简历 ===")
        print(f"文件名: {filename}")
        
        if not filename:
            print("错误：未提供文件名")
            return jsonify({'success': False, 'error': '未提供文件名'})
        
        # 获取文件路径
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"文件路径: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path}")
            return jsonify({'success': False, 'error': '文件不存在'})
        
        # 提取文本
        if file_path.lower().endswith('.pdf'):
            text = extract_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            text = extract_docx(file_path)
        else:
            return jsonify({'success': False, 'error': '不支持的文件类型'})
            
        if not text.strip():
            return jsonify({'success': False, 'error': '文件内容为空'})
            
        # 使用规则解析器
        result, error = rule_parser(text, filename)
        if error:
            print(f"解析出错: {error}")
            return jsonify({'success': False, 'error': error})
            
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        print(f"发生异常: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/jobseeker/bert_parse', methods=['POST'])
def bert_parse_resume():
    """使用BERT模型解析简历API"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        print(f"\n=== 开始BERT解析简历 ===")
        print(f"文件名: {filename}")
        
        if not filename:
            print("错误：未提供文件名")
            return jsonify({'success': False, 'error': '未提供文件名'})
        
        # 获取文件路径
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"文件路径: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path}")
            return jsonify({'success': False, 'error': '文件不存在'})
        
        # 提取文本
        if file_path.lower().endswith('.pdf'):
            text = extract_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            text = extract_docx(file_path)
        else:
            return jsonify({'success': False, 'error': '不支持的文件类型'})
            
        if not text.strip():
            return jsonify({'success': False, 'error': '文件内容为空'})
            
        # 使用BERT解析器
        result, error = bert_parser(text, filename)
        if error:
            print(f"解析出错: {error}")
            return jsonify({'success': False, 'error': error})
            
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        print(f"发生异常: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recruiter/match', methods=['POST'])
def match_talent():
    """匹配人才"""
    try:
        # 检查用户是否登录
        if 'user_id' not in session or session.get('user_type') != 'recruiter':
            return jsonify({'success': False, 'error': '请先登录'}), 401
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
            
        # 获取岗位信息
        job_title = data.get('title', '')
        job_description = data.get('description', '')
        job_requirements = data.get('requirements', '')
        
        if not job_title or not job_description:
            return jsonify({'success': False, 'error': '职位名称和描述不能为空'}), 400
            
        # 获取所有简历
        resumes = list(db.resumes.find())
                    
        if not resumes:
            return jsonify({'success': False, 'error': '未找到任何简历'}), 404
            
        # 使用匹配系统进行匹配
        matcher = Match()
        jaccard_matcher = JaccardMatch()
        job_data = {
            'title': job_title,
            'description': job_description,
            'requirements': job_requirements
        }
        
        matches = []
        jaccard_matches = []
        jaccard_temp = []
        for resume in resumes:
            try:
                processed_resume = matcher.preprocess_resume(resume['data'])
                processed_job = matcher.preprocess_job(job_data)
                score, report = matcher.calculate_similarity(processed_resume, processed_job)
                
                jaccard_resume = resume['data']
                jaccard_score = jaccard_matcher.jaccard_similarity(
                    jaccard_matcher.preprocess_resume(jaccard_resume),
                    jaccard_matcher.preprocess_job(job_data)
                )
                
                education = ''
                experience = ''
                for block in resume['data'].get('blocks', []):
                    if '教育' in block.get('category', ''):
                        education = block.get('content', '')
                    elif '工作' in block.get('category', '') or '项目' in block.get('category', ''):
                        experience = block.get('content', '')
                
                # 提取匹配的关键词
                keywords = []
                for item in report:
                    similarity = float(item['相似度'])
                    if similarity > 0.5:
                        keywords.append(item['模块'].split(' vs ')[0])
                
                matches.append({
                    'resume_id': str(resume['_id']),
                    'name': resume['data'].get('name', '未知'),
                    'similarity': round(float(score * 100), 2),
                    'education': education[:50] + '...' if len(education) > 50 else education,
                    'keywords': keywords[:5],
                    'experience': experience[:80] + '...' if len(experience) > 80 else experience,
                    'filename': resume['data'].get('filename', ''),
                    'jaccard_similarity': round(float(jaccard_score * 100), 2)
                })
                jaccard_temp.append({
                    'resume_id': str(resume['_id']),
                    'name': resume['data'].get('name', '未知'),
                    'similarity': round(float(jaccard_score * 100), 2),
                    'education': education[:50] + '...' if len(education) > 50 else education,
                    'experience': experience[:80] + '...' if len(experience) > 80 else experience,
                    'filename': resume['data'].get('filename', '')
                })
            except Exception as e:
                print(f"处理简历 {str(resume['_id'])} 时出错: {str(e)}")
                continue
        # 按匹配度排序
        matches = [m for m in matches if m['similarity'] > 40.0]
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        if len(matches) > 3:
            matches = matches[:3]
        jaccard_temp.sort(key=lambda x: x['similarity'], reverse=True)
        jaccard_matches = jaccard_temp[:3]
        
        return jsonify({
            'success': True,
            'data': {
                'job_title': job_title,
                'total_matches': len(matches) if matches else 0,
                'matches': matches if isinstance(matches, list) else [],
                'jaccard_matches': jaccard_matches if isinstance(jaccard_matches, list) else []
            }
        })
    
    except Exception as e:
        print(f"匹配过程中发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/jobseeker/upload_to_db', methods=['POST'])
def upload_to_db():
    """上传简历到数据库API"""
    if 'user_id' not in session or session.get('user_type') not in ['jobseeker', 'admin']:
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    data = request.get_json()
    parse_data = data.get('data')
    if not data:
        return jsonify({'success': False, 'error': '无效的请求数据'}), 400
    parse_data = data.get('data')
    if not parse_data:
        return jsonify({'success': False, 'error': '缺少必要参数'}), 400
    # 检查用户是否已有简历（仅对求职者限制，管理员不限制）
    if session.get('user_type') == 'jobseeker':
        existing_resume = db.resumes.find_one({'user_id': session['user_id']})
    else:
        existing_resume = None
    resume_data = {
        'user_id': session['user_id'],
        'data': parse_data,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    if existing_resume:
        db.resumes.update_one(
            {'user_id': session['user_id']},
            {'$set': resume_data}
        )
        message = '简历已更新'
    else:
        resume_data['created_at'] = resume_data['updated_at']
        db.resumes.insert_one(resume_data)
        message = '简历已保存'
    return jsonify({
        'success': True,
        'message': message
    })

@app.route('/api/jobseeker/recommend_jobs', methods=['POST'])
def recommend_jobs():
    """获取岗位推荐API"""
    try:  
        # 1. 检查用户登录状态
        if 'user_id' not in session or session.get('user_type') != 'jobseeker':
            print("错误：用户未登录或不是求职者")
            return jsonify({'success': False, 'error': '请先登录'}), 401
            
        # 2. 获取简历数据
        data = request.get_json()
        
        if not data or 'resume_data' not in data:
            print("错误：缺少简历数据")
            return jsonify({'success': False, 'error': '缺少简历数据'}), 400
            
        resume_data = data['resume_data']
        
        # 3. 读取所有职位数据
        jobs_data = list(db.jobs.find({
            'audit_status': 'approved',
            'status': 'active'
        }))
        print(f"成功读取 {len(jobs_data)} 个职位数据")
        
        if not jobs_data:
            print("错误：没有可用的职位数据")
            return jsonify({'success': False, 'error': '没有可用的职位数据'}), 404
        
        # 4. 初始化推荐系统并生成推荐
        try:
            matcher = Match()
            jaccard_matcher = JaccardMatch()
            
            recommendations = matcher.recommend_jobs(resume_data, jobs_data)
            # Jaccard推荐
            jaccard_raw = jaccard_matcher.recommend_jobs(resume_data, jobs_data, top_k=len(jobs_data))
            # 匹配度保留两位小数，取top3
            jaccard_recommendations = sorted([
                {
                    **item,
                    'score': round(float(item.get('score', 0)), 2),
                    'details': [{
                        **d,
                        '相似度': round(float(d.get('相似度', 0)), 2),
                        '贡献值': round(float(d.get('贡献值', 0)), 2)
                    } for d in item.get('details', [])]
                }
                for item in jaccard_raw
            ], key=lambda x: x['score'], reverse=True)[:3]
            
            print(f"成功生成 {len(recommendations)} 个推荐结果, Jaccard推荐{len(jaccard_recommendations)}个")
            response_data = {
                'success': True,
                'data': {
                    'resume_name': resume_data.get('name', '未知'),
                    'recommendations': recommendations,
                    'jaccard_recommendations': jaccard_recommendations
                }
            }
            
            return app.response_class(
                response=json.dumps(response_data, cls=NumpyEncoder),
                status=200,
                mimetype='application/json'
            )
            
        except Exception as e:
            print(f"推荐过程发生错误: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
    except Exception as e:
        print(f"处理推荐请求时发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobseeker/jobs', methods=['POST'])
def get_jobs():
    """获取岗位列表，支持分页和筛选"""
    try:
        # 检查用户登录状态
        if 'user_id' not in session or session.get('user_type') != 'jobseeker':
            return jsonify({'success': False, 'error': '请先登录'}), 401
            
        # 获取请求参数
        data = request.get_json()
        page = data.get('page', 1)
        keyword = data.get('keyword', '')
        
        # 构建查询条件
        query = {
            'status': 'active',
            'audit_status': 'approved'  # 只返回审核通过的岗位
        }
        if keyword:
            job_or = [
                {'title': {'$regex': keyword, '$options': 'i'}},
                {'description': {'$regex': keyword, '$options': 'i'}},
                {'requirements': {'$regex': keyword, '$options': 'i'}}
            ]
            # 查公司名（用户名）
            recruiter_ids = []
            for user in db.users.find({'username': {'$regex': keyword, '$options': 'i'}}):
                recruiter_ids.append(user['_id'])
            if recruiter_ids:
                job_or.append({'recruiter_id': {'$in': recruiter_ids}})
            query['$or'] = job_or
        
        # 读取所有符合条件的职位数据
        jobs_data = list(db.jobs.find(query))
        
        # 处理每个职位的数据
        processed_jobs = []
        for job in jobs_data:
            # 获取发布者信息
            recruiter = db.users.find_one({'_id': ObjectId(job['recruiter_id'])})
            recruiter_name = recruiter.get('username', '未知') if recruiter else '未知'
            
            # 格式化职位数据
            job_info = {
                '_id': str(job['_id']),
                'title': job.get('title', ''),
                'description': job.get('description', ''),
                'requirements': job.get('requirements', ''),
                'recruiter_name': recruiter_name,
                'created_at': job.get('created_at', ''),
                'matchRate': 0.0  # 初始匹配度为0，实际匹配度由推荐接口计算
            }
            processed_jobs.append(job_info)
        
        # 分页处理
        page_size = 10
        total_jobs = len(processed_jobs)
        total_pages = (total_jobs + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_jobs)
        
        paginated_jobs = processed_jobs[start_idx:end_idx]
        
        return jsonify({
            'success': True,
            'jobs': paginated_jobs,
            'totalJobs': total_jobs,
            'currentPage': page,
            'totalPages': total_pages
        })
        
    except Exception as e:
        print(f"获取岗位列表失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobseeker/job/<job_id>')
def get_job_detail(job_id):
    """获取岗位详情（求职者视角）"""
    print("\n=== 开始获取岗位详情 ===")
    print(f"请求的job_id: {job_id}")
    
    if not ObjectId.is_valid(job_id):
        print(f"无效的ObjectId: {job_id}")
        return jsonify({'success': False, 'error': '无效的岗位ID'}), 400
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401

    try:
        # 查找对应的职位
        job = db.jobs.find_one({
            '_id': ObjectId(job_id),
            'status': 'active',  # 只能查看激活状态的岗位
            'audit_status': 'approved'  # 只能查看审核通过的岗位
        })
        
        if not job:
            return jsonify({'success': False, 'error': '岗位不存在、已关闭或未通过审核'}), 404
        
        # 获取发布者信息
        recruiter = db.users.find_one({'_id': ObjectId(job['recruiter_id'])})
        recruiter_name = recruiter.get('username', '未知') if recruiter else '未知'
        
        # 处理日期格式
        created_at = job.get('created_at', '')
        if isinstance(created_at, datetime):
            created_at = created_at.strftime('%Y-%m-%d %H:%M:%S')
        
        # 检查是否已申请
        has_applied = db.applications.find_one({
            'job_id': ObjectId(job_id),
            'user_id': ObjectId(session['user_id'])
        }) is not None
        
        # 格式化岗位数据
        job_data = {
            '_id': str(job['_id']),
            'title': job.get('title', ''),
            'description': job.get('description', ''),
            'requirements': job.get('requirements', ''),
            'recruiter_name': recruiter_name,
            'created_at': created_at,
            'has_applied': has_applied,
            'audit_status': job.get('audit_status', 'pending'),
            'audit_comment': job.get('audit_comment', '')
        }
        
        return jsonify({
            'success': True,
            'job': job_data
        })
    except Exception as e:
        print(f"获取岗位详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobseeker/apply', methods=['POST'])
def apply_job():
    """申请职位"""
    try:
        # 检查用户登录状态
        if 'user_id' not in session or session.get('user_type') != 'jobseeker':
            return jsonify({'success': False, 'error': '请先登录'}), 401

        job_id = request.form.get('jobId')
        note = request.form.get('note', '')
        resume_file = request.files.get('resume')
        resume_file_name = request.form.get('resumeFileName')
            
        if not job_id:
            return jsonify({'success': False, 'error': '缺少岗位ID'}), 400
            
        # 检查职位是否存在且处于激活状态
        job = db.jobs.find_one({
            '_id': ObjectId(job_id),
            'status': 'active'
        })
        if not job:
            return jsonify({'success': False, 'error': '职位不存在或已关闭'}), 404
            
        # 检查是否已经申请过
        existing_application = db.applications.find_one({
            'job_id': ObjectId(job_id),
            'user_id': ObjectId(session['user_id'])
        })
        if existing_application:
            return jsonify({'success': False, 'error': '您已经申请过这个职位了'}), 400
            
        # 1. 优先使用resumeFileName
        if resume_file_name:
            # 检查文件是否存在
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file_name)
            if not os.path.exists(filepath):
                return jsonify({'success': False, 'error': '简历文件不存在，请重新上传'}), 400
            filename = resume_file_name
        # 2. 兼容老方式：上传文件
        elif resume_file:
            if resume_file.filename == '':
                return jsonify({'success': False, 'error': '请选择要上传的简历'}), 400
            if not allowed_file(resume_file.filename):
                return jsonify({'success': False, 'error': '只支持PDF、DOCX格式的文件'}), 400
            filename = secure_filename(resume_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(filepath)
        else:
            return jsonify({'success': False, 'error': '请先上传简历'}), 400
        
        # 保存申请记录
        application_data = {
            'job_id': ObjectId(job_id),
            'user_id': ObjectId(session['user_id']),
            'resume_filename': filename,
            'note': note,
            'status': 'pending',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        db.applications.insert_one(application_data)
        return jsonify({'success': True, 'message': '申请成功'})
    except Exception as e:
        print(f"申请职位失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recruiter/my_jobs')
def get_my_jobs():
    """获取招聘者发布的岗位列表"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 从MongoDB读取当前用户发布的所有岗位
        jobs = list(db.jobs.find({'recruiter_id': ObjectId(session['user_id'])}))
        
        # 处理数据格式
        jobs_data = []
        for job in jobs:
            # 处理日期格式
            created_at = job.get('created_at', '')
            if isinstance(created_at, datetime):
                created_at = created_at.strftime('%Y-%m-%d %H:%M:%S')
                
            # 处理审核时间
            audit_time = job.get('audit_time', '')
            if isinstance(audit_time, datetime):
                audit_time = audit_time.strftime('%Y-%m-%d %H:%M:%S')
                
            job_data = {
                '_id': str(job['_id']),
                'title': job.get('title', ''),
                'description': job.get('description', '')[:150] + '...' if len(job.get('description', '')) > 150 else job.get('description', ''),
                'requirements': job.get('requirements', ''),
                'created_at': created_at,
                'status': job.get('status', 'active'),
                'audit_status': job.get('audit_status', 'pending'),  # 添加审核状态
                'audit_comment': job.get('audit_comment', ''),  # 添加审核意见
                'audit_time': audit_time,  # 添加审核时间
                'applications_count': db.applications.count_documents({'job_id': ObjectId(str(job['_id']))})
            }
            jobs_data.append(job_data)
        
        # 按创建时间倒序排序（使用字符串比较）
        jobs_data.sort(key=lambda x: x['created_at'] or '', reverse=True)
        
        return jsonify({
            'success': True,
            'jobs': jobs_data
        })
    except Exception as e:
        print(f"获取招聘者岗位列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recruiter/job/<job_id>')
def get_recruiter_job_detail(job_id):
    """获取招聘者发布的某个岗位详情"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 查找对应的岗位
        job = db.jobs.find_one({
            '_id': ObjectId(job_id),
            'recruiter_id': ObjectId(session['user_id'])  # 确保使用ObjectId
        })
        
        if not job:
            return jsonify({'success': False, 'error': '岗位不存在或无权访问'}), 404
            
        # 格式化岗位数据
        job_data = {
            '_id': str(job['_id']),
            'title': job.get('title', ''),
            'description': job.get('description', ''),
            'requirements': job.get('requirements', ''),
            'created_at': job.get('created_at', ''),
            'status': job.get('status', 'active')
        }
        
        return jsonify({
            'success': True,
            'job': job_data
        })
    except Exception as e:
        print(f"获取岗位详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recruiter/job/<job_id>/status', methods=['POST'])
def update_job_status(job_id):
    """更新岗位状态（开启/关闭）"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        data = request.get_json()
        new_status = data.get('status')
        if new_status not in ['active', 'closed']:
            return jsonify({'success': False, 'error': '无效的状态值'}), 400
        update_fields = {'status': new_status}
        # 如果是重新发布（active），重置审核状态和审核意见
        if new_status == 'active':
            update_fields['audit_status'] = 'pending'
            update_fields['audit_comment'] = ''
            update_fields['audit_time'] = None
        result = db.jobs.update_one(
            {
                '_id': ObjectId(job_id),
                'recruiter_id': ObjectId(session['user_id'])
            },
            {'$set': update_fields}
        )
        if result.matched_count == 0:
            return jsonify({'success': False, 'error': '岗位不存在或无权操作'}), 404
        return jsonify({
            'success': True,
            'message': '岗位状态已更新'
        })
    except Exception as e:
        print(f"更新岗位状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recruiter/job/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """删除岗位"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 删除岗位
        result = db.jobs.delete_one({
            '_id': ObjectId(job_id),
            'recruiter_id': ObjectId(session['user_id']) 
        })
        
        if result.deleted_count == 0:
            return jsonify({'success': False, 'error': '岗位不存在或无权删除'}), 404
            
        # 同时删除相关的申请记录
        db.applications.delete_many({'job_id': ObjectId(job_id)})  
        
        return jsonify({
            'success': True,
            'message': '岗位已删除'
        })
    except Exception as e:
        print(f"删除岗位失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
            
        # 获取注册信息
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        user_type = data.get('userType')
        
        # 验证必填字段
        if not all([username, email, password, user_type]):
            return jsonify({'success': False, 'error': '请填写所有必填字段'}), 400
            
        # 检查用户名和邮箱是否已存在
        existing_user = db.users.find_one({
            'userType': user_type,
            '$or': [
                {'username': username},
                {'email': email}
            ]
        })
        
        if existing_user:
            if existing_user.get('username') == username:
                return jsonify({'success': False, 'error': '用户名已存在'}), 400
            else:
                return jsonify({'success': False, 'error': '邮箱已被注册'}), 400
            
        # 创建新用户
        user_data = {
            'username': username,
            'email': email,
            'password': password,  
            'userType': user_type,
            'status': 'active',  
            'createdAt': datetime.now(),
            'lastLogin': datetime.now(),
            'updatedAt': datetime.now()  
        }
        
        # 保存到MongoDB
        result = db.users.insert_one(user_data)
        
        return jsonify({
            'success': True,
            'message': '注册成功',
            'userId': str(result.inserted_id)
        })
        
    except Exception as e:
        print(f"注册过程发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin')
def admin():
    """管理员后台页面路由"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return redirect(url_for('index'))
    return render_template('admin.html')

@app.route('/api/admin/stats')
def admin_stats():
    """获取系统统计数据API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 获取统计数据
        stats = {
            'users': db.users.count_documents({}),
            'resumes': db.resumes.count_documents({}),
            'jobs': db.jobs.count_documents({})
        }
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/users')
def admin_users():
    """获取用户列表API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        users = list(db.users.find())
        # 转换ObjectId为字符串，处理日期格式
        for user in users:
            user['_id'] = str(user['_id'])
            user['createdAt'] = user['createdAt'].strftime('%Y-%m-%d %H:%M:%S') if user.get('createdAt') else ''
            user['lastLogin'] = user['lastLogin'].strftime('%Y-%m-%d %H:%M:%S') if user.get('lastLogin') else ''
            user['updatedAt'] = user['updatedAt'].strftime('%Y-%m-%d %H:%M:%S') if user.get('updatedAt') else ''
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        print(f"获取用户列表失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/users/<user_id>', methods=['DELETE'])
def admin_delete_user(user_id):
    """删除用户API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 不允许删除管理员账号
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if user and user.get('userType') == 'admin':
            return jsonify({'success': False, 'error': '不能删除管理员账号'}), 400
            
        # 删除用户
        result = db.users.delete_one({'_id': ObjectId(user_id)})
        if result.deleted_count == 0:
            return jsonify({'success': False, 'error': '用户不存在'}), 404
            
        # 同时删除该用户的简历
        db.resumes.delete_many({'user_id': ObjectId(user_id)})
        # 同时删除该用户发布的岗位
        db.jobs.delete_many({'recruiter_id': ObjectId(user_id)})
        
        return jsonify({'success': True, 'message': '用户删除成功'})
    except Exception as e:
        print(f"删除用户失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/resumes')
def admin_resumes():
    """获取简历列表API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        resumes = list(db.resumes.find())
        # 获取用户名并处理日期格式
        for resume in resumes:
            user = db.users.find_one({'_id': ObjectId(resume['user_id'])})
            resume['username'] = user.get('username', '未知用户') if user else '未知用户'
            resume['_id'] = str(resume['_id'])
            resume['created_at'] = resume.get('created_at', '')
            resume['updated_at'] = resume.get('updated_at', '')
        return jsonify({'success': True, 'resumes': resumes})
    except Exception as e:
        print(f"获取简历列表失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/resumes/<resume_id>', methods=['DELETE'])
def admin_delete_resume(resume_id):
    """删除简历API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        db.resumes.delete_one({'_id': ObjectId(resume_id)})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/jobs')
def admin_jobs():
    """获取岗位列表API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        jobs = list(db.jobs.find())
        # 获取发布者用户名并处理日期格式
        for job in jobs:
            recruiter = db.users.find_one({'_id': job['recruiter_id']})
            job['recruiter_name'] = recruiter.get('username', '未知用户') if recruiter else '未知用户'
            job['_id'] = str(job['_id'])
            job['recruiter_id'] = str(job['recruiter_id'])
            if isinstance(job.get('created_at'), datetime):
                job['created_at'] = job['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                job['created_at'] = ''
            # 添加审核状态显示
            job['audit_status'] = job.get('audit_status', 'pending')
            job['audit_comment'] = job.get('audit_comment', '')
            if isinstance(job.get('audit_time'), datetime):
                job['audit_time'] = job['audit_time'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                job['audit_time'] = ''
        return jsonify({'success': True, 'jobs': jobs})
    except Exception as e:
        print(f"获取岗位列表失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/jobs/<job_id>', methods=['DELETE'])
def admin_delete_job(job_id):
    """删除岗位API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        db.jobs.delete_one({'_id': ObjectId(job_id)})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recruiter/job/<job_id>', methods=['GET'])
def get_job(job_id):
    """获取岗位详情API"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        job = db.jobs.find_one({
            '_id': ObjectId(job_id),
            'recruiter_id': ObjectId(session['user_id'])
        })
        
        if not job:
            return jsonify({'success': False, 'error': '岗位不存在或无权查看'}), 404
            
        return jsonify({
            'success': True,
            'job': {
                '_id': str(job['_id']),
                'title': job.get('title', ''),
                'description': job.get('description', ''),
                'requirements': job.get('requirements', ''),
                'status': job.get('status', 'active'),
                'created_at': job.get('created_at', ''),
                'updated_at': job.get('updated_at', '')
            }
        })
    except Exception as e:
        print(f"获取岗位详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/resumes/<resume_id>', methods=['GET'])
def get_resume_detail(resume_id):
    """获取单个简历详情"""
    try:
        # 验证管理员权限
        if 'user_id' not in session or session.get('user_type') != 'admin':
                return jsonify({
                    'success': False, 
                    'error': '未登录或权限不足'
                    }), 401
        
        # 从数据库获取简历信息
        resume = db.resumes.find_one({'_id': ObjectId(resume_id)})
        if not resume:
            return jsonify({
                'success': False,
                'error': '简历不存在'
            }), 404
        
        # 处理ObjectId为可序列化格式
        resume['_id'] = str(resume['_id'])
        
        return jsonify({
            'success': True,
            'resume': resume
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobseeker/profile', methods=['GET'])
def get_jobseeker_profile():
    """获取求职者基本信息"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user:
            return jsonify({'success': False, 'error': '用户不存在'}), 404
        # 增加手机号字段
        user_data = {
            'username': user.get('username', ''),
            'email': user.get('email', ''),
            'phone': user.get('phone', ''),
            'is_subscribed': user.get('is_subscribed', False),
            'subscribe_expire': user.get('subscribe_expire').strftime('%Y-%m-%d %H:%M:%S') if user.get('subscribe_expire') else '',
            'subscribe_type': user.get('subscribe_type', '')
        }
        return jsonify({
            'success': True,
            'user': user_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobseeker/bind_phone', methods=['POST'])
def bind_jobseeker_phone():
    """绑定手机号"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        data = request.get_json()
        phone = data.get('phone', '').strip()
        if not phone or not phone.isdigit() or len(phone) != 11:
            return jsonify({'success': False, 'error': '手机号格式不正确'}), 400
        db.users.update_one({'_id': ObjectId(session['user_id'])}, {'$set': {'phone': phone}})
        return jsonify({'success': True})
    except Exception as e:
        print(f"绑定手机号失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobseeker/resumes', methods=['GET'])
def get_jobseeker_resumes():
    """获取求职者的简历列表API"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        print('[简历列表] 未登录或权限不足')
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        user_id = session['user_id']
        print(f'[简历列表] 当前用户user_id: {user_id}')
        query = {'user_id': user_id}
        print(f'[简历列表] 查询条件: {query}')
        resumes = list(db.resumes.find(query))
        print(f'[简历列表] 查到简历数量: {len(resumes)}')
        for idx, resume in enumerate(resumes):
            print(f'  [简历{idx}] _id: {resume.get("_id")}, data.name: {resume.get("data", {}).get("name")}, data.filename: {resume.get("data", {}).get("filename")}, created_at: {resume.get("created_at")}, updated_at: {resume.get("updated_at")}')
        resumes_data = []
        for resume in resumes:
            resumes_data.append({
                '_id': str(resume['_id']),
                'filename': resume.get('data', {}).get('filename', ''),
                'name': resume.get('data', {}).get('name', '未命名简历'),
                'created_at': resume.get('created_at', ''),
                'updated_at': resume.get('updated_at', '')
            })
        return jsonify({
            'success': True,
            'resumes': resumes_data
        })
    except Exception as e:
        print(f"[简历列表] 获取简历列表失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobseeker/applications', methods=['GET'])
def get_jobseeker_applications():
    """获取求职者的投递记录"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 获取用户的所有投递记录
        applications = list(db.applications.find({'user_id': ObjectId(session['user_id'])}))
        
        # 格式化投递记录数据
        applications_data = []
        for app in applications:
            # 获取对应的职位信息
            job = db.jobs.find_one({'_id': ObjectId(app['job_id'])})
            # 获取公司信息
            company = db.users.find_one({'_id': ObjectId(job['recruiter_id'])}) if job else None
            
            app_data = {
                '_id': str(app['_id']),
                'job_id': str(app['job_id']),
                'job_title': job.get('title', '未知职位') if job else '未知职位',
                'company': company.get('company_name', '未知公司') if company else '未知公司',
                'status': app.get('status', 'pending'),
                'created_at': app.get('created_at', '').strftime('%Y-%m-%d %H:%M:%S') if app.get('created_at') else '',
                'comment': app.get('comment', '')  
            }
            applications_data.append(app_data)
        
        return jsonify({
            'success': True,
            'applications': applications_data
        })
    except Exception as e:
        print(f"获取投递记录失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobseeker/resumes/<resume_id>', methods=['DELETE'])
def delete_jobseeker_resume(resume_id):
    """删除求职者的简历"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 检查简历是否存在且属于当前用户
        resume = db.resumes.find_one({
            '_id': ObjectId(resume_id),
            'user_id': session['user_id']  
        })
        
        if not resume:
            return jsonify({'success': False, 'error': '简历不存在或无权访问'}), 404
        
        # 删除数据库记录
        db.resumes.delete_one({'_id': ObjectId(resume_id)})
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"删除简历失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/jobseeker_profile')
def jobseeker_profile():
    """求职者个人中心页面"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return redirect('/')
    return render_template('jobseeker_profile.html')

@app.route('/api/account/delete', methods=['POST'])
def delete_account():
    """账户注销，支持求职者和招聘者"""
    if 'user_id' not in session or session.get('user_type') not in ['jobseeker', 'recruiter']:
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        user_id = session['user_id']
        user_type = session['user_type']
        # 删除用户主表
        db.users.delete_one({'_id': ObjectId(user_id)})
        # 删除相关数据
        if user_type == 'jobseeker':
            db.resumes.delete_many({'user_id': ObjectId(user_id)})
            db.applications.delete_many({'user_id': ObjectId(user_id)})
        elif user_type == 'recruiter':
            # 删除招聘者发布的岗位及相关申请
            jobs = list(db.jobs.find({'recruiter_id': ObjectId(user_id)}))
            for job in jobs:
                db.applications.delete_many({'job_id': job['_id']})
            db.jobs.delete_many({'recruiter_id': ObjectId(user_id)})
        session.clear()
        return jsonify({'success': True, 'message': '账户已注销'})
    except Exception as e:
        print(f"账户注销失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobseeker/resume_parse_detail_api/<resume_id>')
def resume_parse_detail_api(resume_id):
    """返回简历解析内容JSON，供前端模态框展示"""
    if 'user_id' not in session or session.get('user_type') != 'jobseeker':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    from bson.objectid import ObjectId
    resume = db.resumes.find_one({'_id': ObjectId(resume_id), 'user_id': session['user_id']})
    if not resume:
        return jsonify({'success': False, 'error': '简历不存在或无权查看'}), 404
    parse_data = resume.get('data', {})
    return jsonify({'success': True, 'parse': parse_data})

@app.route('/api/recruiter/resume_detail/<resume_id>')
def recruiter_resume_detail(resume_id):
    """招聘者查看简历详情"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    from bson.objectid import ObjectId
    resume = db.resumes.find_one({'_id': ObjectId(resume_id)})
    if not resume:
        return jsonify({'success': False, 'error': '简历不存在'}), 404
    parse_data = resume.get('data', {})
    return jsonify({'success': True, 'parse': parse_data})

@app.route('/api/recruiter/view_resume/<filename>')
def recruiter_view_resume(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            if filename.lower().endswith('.pdf'):
                return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')
            elif filename.lower().endswith('.docx'):
                return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            else:
                return jsonify({
                    'success': False,
                    'error': '不支持的文件类型'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': '文件不存在'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/feedback/recommend', methods=['POST'])
def feedback_recommend():
    """推荐/人才检索整体评分反馈API"""
    if 'user_id' not in session or session.get('user_type') not in ['jobseeker', 'recruiter']:
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        data = request.get_json()
        user_id = session['user_id']
        user_type = session['user_type']
        recommend_list = data.get('recommend_list', [])
        scene = data.get('scene', '')
        score = int(data.get('score', 0))
        comment = data.get('comment', '').strip()
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not recommend_list or not isinstance(recommend_list, list) or scene not in ['job_recommend', 'talent_match']:
            return jsonify({'success': False, 'error': '参数不完整'}), 400

        exist = db.recommend_feedback.find_one({
            'user_id': user_id,
            'scene': scene,
            'recommend_list': recommend_list
        })
        if exist:
            return jsonify({'success': False, 'error': '您已评价过本次推荐'}), 400

        feedback = {
            'user_id': user_id,
            'user_type': user_type,
            'recommend_list': recommend_list,
            'score': score,
            'comment': comment,
            'scene': scene,
            'created_at': created_at
        }
        db.recommend_feedback.insert_one(feedback)
        return jsonify({'success': True, 'message': '反馈已提交'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/recommend_feedback', methods=['GET'])
def admin_recommend_feedback():
    """管理员查看所有推荐评分反馈"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        # 分页参数
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        skip = (page - 1) * page_size
        
        total = db.recommend_feedback.count_documents({})
        feedbacks = list(db.recommend_feedback.find().skip(skip).limit(page_size).sort('created_at', -1))
        
        # 格式化
        for fb in feedbacks:
            fb['_id'] = str(fb['_id'])
            
        return jsonify({
            'success': True, 
            'total': total, 
            'page': page, 
            'page_size': page_size, 
            'feedbacks': feedbacks
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/recruiter_profile')
def recruiter_profile():
    """招聘者个人中心页面"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return redirect('/')
    return render_template('recruiter_profile.html')

@app.route('/api/recruiter/profile', methods=['GET'])
def get_recruiter_profile():
    """获取招聘者基本信息"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user:
            return jsonify({'success': False, 'error': '用户不存在'}), 404
        user_data = {
            'username': user.get('username', ''),
            'email': user.get('email', ''),
            'phone': user.get('phone', ''),
            'company_name': user.get('company_name', ''),
            'is_subscribed': user.get('is_subscribed', False),
            'subscribe_expire': user.get('subscribe_expire').strftime('%Y-%m-%d %H:%M:%S') if user.get('subscribe_expire') else '',
            'subscribe_type': user.get('subscribe_type', '')
        }
        return jsonify({'success': True, 'user': user_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recruiter/bind_phone', methods=['POST'])
def bind_recruiter_phone():
    """绑定手机号"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        data = request.get_json()
        phone = data.get('phone', '').strip()
        if not phone or not phone.isdigit() or len(phone) != 11:
            return jsonify({'success': False, 'error': '手机号格式不正确'}), 400
        db.users.update_one({'_id': ObjectId(session['user_id'])}, {'$set': {'phone': phone}})
        return jsonify({'success': True})
    except Exception as e:
        print(f"绑定手机号失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recruiter/update_company', methods=['POST'])
def update_company():
    """更新公司名称"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        data = request.get_json()
        company_name = data.get('company_name', '').strip()
        if not company_name:
            return jsonify({'success': False, 'error': '公司名称不能为空'}), 400
        db.users.update_one({'_id': ObjectId(session['user_id'])}, {'$set': {'company_name': company_name}})
        return jsonify({'success': True})
    except Exception as e:
        print(f"更新公司名称失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recruiter/job_applications', methods=['GET'])
def get_recruiter_job_applications():
    """获取招聘者发布的岗位及其投递记录"""
    if 'user_id' not in session or session.get('user_type') != 'recruiter':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    
    try:
        # 获取该招聘者的所有岗位
        jobs = list(db.jobs.find({'recruiter_id': ObjectId(session['user_id'])}))
        
        # 获取每个岗位的投递记录
        jobs_with_applications = []
        for job in jobs:
            applications = list(db.applications.find({'job_id': job['_id']}))
            
            # 如果岗位有投递记录，则添加到结果中
            if applications:
                # 获取每个申请的投递者信息并处理ObjectId
                formatted_applications = []
                for app in applications:
                    user = db.users.find_one({'_id': ObjectId(app['user_id'])})
                    formatted_app = {
                        '_id': str(app['_id']),
                        'user_id': str(app['user_id']),
                        'job_id': str(app['job_id']),
                        'user_name': user.get('username', '未知用户') if user else '未知用户',
                        'resume_filename': app.get('resume_filename', ''),
                        'note': app.get('note', ''),
                        'status': app.get('status', 'pending'),
                        'comment': app.get('comment', ''),
                        'created_at': app.get('created_at', '').strftime('%Y-%m-%d %H:%M:%S') if app.get('created_at') else '',
                        'updated_at': app.get('updated_at', '').strftime('%Y-%m-%d %H:%M:%S') if app.get('updated_at') else ''
                    }
                    formatted_applications.append(formatted_app)
                
                jobs_with_applications.append({
                    '_id': str(job['_id']),
                    'title': job.get('title', ''),
                    'applications': formatted_applications
                })
        
        return jsonify({
            'success': True,
            'jobs': jobs_with_applications
        })
        
    except Exception as e:
        print(f"获取投递记录失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/applications')
def admin_get_applications():
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    apps = list(db.db['applications'].find())
    result = []
    for app in apps:
        user = db.users.find_one({'_id': ObjectId(app['user_id'])})
        job = db.jobs.find_one({'_id': ObjectId(app['job_id'])})
        company = db.users.find_one({'_id': job['recruiter_id']}) if job else None
        result.append({
            '_id': str(app['_id']),
            'user_name': user.get('username', '未知') if user else '未知',
            'job_title': job.get('title', '未知') if job else '未知',
            'company': company.get('company_name', '未知') if company else '未知',
            'status': app.get('status', ''),
            'comment': app.get('comment', ''),
            'created_at': app.get('created_at', '').strftime('%Y-%m-%d %H:%M:%S') if app.get('created_at') else ''
        })
    return jsonify({'success': True, 'applications': result})

@app.route('/api/admin/job/<job_id>/audit', methods=['POST'])
def admin_audit_job_status(job_id):
    """管理员修改岗位审核状态"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    data = request.get_json() or {}
    status = data.get('status')
    comment = data.get('comment', '')
    if status not in ['approved', 'pending', 'rejected']:
        return jsonify({'success': False, 'error': '无效的状态'}), 400
    update_data = {
        'audit_status': status,
        'audit_comment': comment,
        'audit_time': datetime.now()
    }
    update_data['status'] = 'active' if status == 'approved' else 'inactive'
    result = db.jobs.update_one({'_id': ObjectId(job_id)}, {'$set': update_data})
    if result.modified_count == 0:
        return jsonify({'success': False, 'error': '岗位不存在或未修改'}), 404
    return jsonify({'success': True, 'message': '岗位状态已更新'})

@app.route('/api/admin/application/<application_id>/status', methods=['POST'])
def admin_update_application_status(application_id):
    """管理员修改投递状态"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    data = request.get_json() or {}
    status = data.get('status')
    comment = data.get('comment', '')
    if status not in ['accepted', 'pending', 'rejected']:
        return jsonify({'success': False, 'error': '无效的状态'}), 400
    update_data = {
        'status': status,
        'comment': comment,
        'updated_at': datetime.now()
    }
    result = db.db['applications'].update_one({'_id': ObjectId(application_id)}, {'$set': update_data})
    if result.modified_count == 0:
        return jsonify({'success': False, 'error': '申请记录不存在或未修改'}), 404
    return jsonify({'success': True, 'message': '投递状态已更新'})

@app.route('/subscribe')
def subscribe():
    """订阅页面路由"""
    return render_template('subscribe.html')

@app.route('/api/subscribe', methods=['POST'])
def api_subscribe():
    """处理订阅请求API"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    data = request.get_json() or {}
    sub_type = data.get('type', 'month')
    now = datetime.now()
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    old_expire = user.get('subscribe_expire')
    # 判断是否续订
    if old_expire and isinstance(old_expire, datetime) and old_expire > now:
        base_time = old_expire
    else:
        base_time = now
    if sub_type == 'year':
        expire = base_time + timedelta(days=365)
    elif sub_type == 'week':
        expire = base_time + timedelta(days=7)
    else:
        expire = base_time + timedelta(days=30)
    db.users.update_one({'_id': ObjectId(session['user_id'])}, {'$set': {'is_subscribed': True, 'subscribe_expire': expire, 'subscribe_type': sub_type}})
    return jsonify({'success': True, 'message': '订阅成功', 'expire': expire.strftime('%Y-%m-%d %H:%M:%S')})

@app.route('/api/admin/users/<user_id>/status', methods=['POST'])
def admin_update_user_status(user_id):
    """更新用户状态API"""
    if 'user_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        data = request.get_json() or {}
        status = data.get('status')
        if status not in ['active', 'disabled']:
            return jsonify({'success': False, 'error': '无效的状态'}), 400
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'success': False, 'error': '用户不存在'}), 404
        db.users.update_one({'_id': ObjectId(user_id)}, {'$set': {'status': status}})
        return jsonify({'success': True, 'message': '状态已更新'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 创建管理员用户
def create_admin_user():
    """创建管理员用户"""
    admin = db.users.find_one({'userType': 'admin'})
    if not admin:
        admin_data = {
            'username': 'admin',
            'email': 'admin@example.com',
            'password': 'admin123',
            'userType': 'admin',
            'status': 'active',
            'createdAt': datetime.now(),
            'lastLogin': datetime.now(),
            'updatedAt': datetime.now()
        }
        db.users.insert_one(admin_data)
        print('管理员用户已创建')

@app.route('/api/change_password', methods=['POST'])
def change_password():
    """修改密码API"""
    if 'user_id' not in session or session.get('user_type') not in ['jobseeker', 'recruiter']:
        return jsonify({'success': False, 'error': '未登录或权限不足'}), 401
    try:
        data = request.get_json()
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'error': '请提供当前密码和新密码'}), 400
        
        # 获取用户信息
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user:
            return jsonify({'success': False, 'error': '用户不存在'}), 404
        
        # 验证当前密码
        if user['password'] != current_password: 
            return jsonify({'success': False, 'error': '当前密码错误'}), 400
        
        # 更新密码
        db.users.update_one(
            {'_id': ObjectId(session['user_id'])},
            {'$set': {'password': new_password}}  
        )
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"修改密码失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # create_admin_user()  # 确保管理员用户存在
    app.run(debug=True)
