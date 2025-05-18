from datetime import timedelta

class Config:
    # Flask配置
    SECRET_KEY = 'your-secret-key-here'  # 用于session加密
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)  # 设置session 7天后过期
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max-limit
    UPLOAD_FOLDER = 'static/uploads'
    
    # MongoDB配置
    MONGO_URI = 'mongodb://localhost:27017/'
    MONGO_DBNAME = 'resume_system'
    
    # 集合名称
    COLLECTION_USERS = 'users'
    COLLECTION_RESUMES = 'resumes'
    COLLECTION_JOBS = 'jobs'
    COLLECTION_APPLICATIONS = 'applications'
    COLLECTION_RECOMMEND_FEEDBACK = 'recommend_feedback'
    