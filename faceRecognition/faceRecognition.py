import os
import numpy as np
import json
from PIL import Image
import io
import face_recognition

# 人脸数据库目录路径
FACE_DB_DIR = "static/face_db"
# 确保人脸数据库目录存在，如果不存在则创建
if not os.path.exists(FACE_DB_DIR):
    os.makedirs(FACE_DB_DIR)

# 全局变量，用于存储加载后的已知人脸编码和对应的名称
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    print("开始加载人脸库...")
    print(f"人脸库目录: {os.path.abspath(FACE_DB_DIR)}")
    
    if not os.path.exists(FACE_DB_DIR):
        print(f"错误: 目录不存在 {FACE_DB_DIR}")
        return
    
    person_dirs = os.listdir(FACE_DB_DIR)
    print(f"找到 {len(person_dirs)} 个人物目录")
    
    for person_name in person_dirs:
        person_dir = os.path.join(FACE_DB_DIR, person_name)
        if os.path.isdir(person_dir):
            print(f"加载 {person_name}...")
            image_files = os.listdir(person_dir)
            
            for image_name in image_files:
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, image_name)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_locations = face_recognition.face_locations(image) # hog 模型
                        
                        if face_locations:
                            face_encodings = face_recognition.face_encodings(image, face_locations)
                            if face_encodings:
                                known_face_encodings.append(face_encodings[0]) # 取第一个人脸
                                known_face_names.append(person_name)
                                print(f"  成功加载: {image_name}")
                            else:
                                print(f"  警告: 无法提取特征 {image_name}")
                        else:
                            print(f"  警告: 未检测到人脸 {image_name}")
                    except Exception as e:
                        print(f"  错误处理照片 {image_name}: {str(e)}")
                        # import traceback
                        # print(traceback.format_exc())
    
    print(f"人脸库加载完成，共 {len(known_face_names)} 人脸特征")
    if not known_face_names:
        print("警告: 未加载到任何人脸特征")

def recognize_face_from_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        face_locations = face_recognition.face_locations(image_array)
        if not face_locations:
            return {'success': True, 'faces': [], 'message': '未检测到人脸'}
        
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if not known_face_encodings:
                name = "unknown"
                confidence = 0.0
            else:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.53)
                name = "unknown"
                confidence = 0.0
                
                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index] # 距离转置信度
                        confidence = max(0.0, min(1.0, confidence))
                        if confidence < 0.47: # 低置信度阈值
                            name = "unknown"
                            confidence = 0.0
            
            face_result = {
                'location': [top, right, bottom, left],
                'name': name,
                'confidence': float(confidence)
            }
            faces.append(face_result)
            print(f"识别到: {name}, 置信度: {confidence:.2f}")
        
        result = {'success': True, 'faces': faces, 'message': '识别完成'}
        print("识别结果:", json.dumps(result, ensure_ascii=False, indent=2))
        return result
        
    except Exception as e:
        error_message = f'处理图片时出错: {str(e)}'
        print(f"错误: {error_message}")
        return {'success': False, 'faces': [], 'message': error_message}

def init_face_recognition():
    load_known_faces() 