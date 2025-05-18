from flask import Flask, request, jsonify, render_template, redirect
import faceRecognition
import os
import json

app = Flask(__name__)


@app.route('/api/recognize', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    # 检查文件
    if 'image' not in request.files:
        response_data = {'success': False, 'message': '未上传图片文件'}
        print("错误:", response_data['message'])
        print("响应:", json.dumps(response_data, ensure_ascii=False, indent=2))
        return jsonify(response_data)
    
    file = request.files['image']
    if not file.filename:
        response_data = {'success': False, 'message': '未选择文件'}
        print("错误:", response_data['message'])
        print("响应:", json.dumps(response_data, ensure_ascii=False, indent=2))
        return jsonify(response_data)
    
    # 检查文件扩展名
    allowed_extensions = {'.png', '.jpg'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        response_data = {'success': False, 'message': '无效的图像文件格式'}
        print(f"错误: {response_data['message']} - {file.filename}")
        print("响应:", json.dumps(response_data, ensure_ascii=False, indent=2))
        return jsonify(response_data)
    
    try:
        image_data = file.read()
        result = faceRecognition.recognize_face_from_image(image_data)
        
        # 处理识别结果
        if result.get('success', False):
            if not result.get('faces'):
                print("信息: 未检测到人脸")
                print("响应:", json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("识别完成")
        else:
            print(f"错误: 人脸识别服务失败 - {result.get('message', '未知错误')}")
            print("响应:", json.dumps(result, ensure_ascii=False, indent=2))
            
        return jsonify(result)

    except Exception as e:
        # 处理错误
        response_data = {'success': False, 'message': f'处理请求时发生意外错误: {str(e)}'}
        print(f"错误: {response_data['message']}")
        print("响应:", json.dumps(response_data, ensure_ascii=False, indent=2))
        return jsonify(response_data)


if __name__ == '__main__':
    faceRecognition.init_face_recognition()
    app.run(debug=True)