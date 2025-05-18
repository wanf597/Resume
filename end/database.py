from pymongo import MongoClient
from config import Config

class MongoDB:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance.client = MongoClient(Config.MONGO_URI)
            cls._instance.db = cls._instance.client[Config.MONGO_DBNAME]
        return cls._instance
    
    @property
    def users(self):
        return self.db[Config.COLLECTION_USERS]
    
    @property
    def resumes(self):
        return self.db[Config.COLLECTION_RESUMES]
    
    @property
    def jobs(self):
        return self.db[Config.COLLECTION_JOBS]
    
    @property
    def applications(self):
        return self.db[Config.COLLECTION_APPLICATIONS]
    
    @property
    def recommend_feedback(self):
        return self.db[Config.COLLECTION_RECOMMEND_FEEDBACK]