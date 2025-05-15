from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import tempfile
import uuid
from flask_cors import CORS
from joblib import load
import torch.nn as nn
import numpy as np
import torch
from models.model import seq2seq,Seq2SeqWithMultiHeadAttention
from models.mediapipe_method import process_video_with_mediapipe
from models.utils import allowed_file,ALLOWED_EXTENSIONS,model_dict,list_proLR,left_right_hand
app = Flask(__name__)
CORS(app)
# Configuration

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    """Render the upload form"""
    return render_template('upload.html')


@app.route('/process_video', methods=['POST'])
def process_video():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    model_name = request.form.get('model')
    print(f"Selected model: {model_name}")  # 打印模型名称
    print(model_dict[model_name])
    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the file permanently
        file.save(save_path)
        print(f"File saved to: {save_path}")

        # Verify file exists
        if not os.path.exists(save_path):
            raise FileNotFoundError("File was not saved correctly")

        # Process the video
        landmarks = process_video_with_mediapipe(save_path,sample_interval=model_dict[model_name][2])
        # print(landmarks)
        # 模型参数
        input_dim = model_dict[model_name][3]  # 输入词汇表大小(等于原词汇表大小+2，+2加的是结束符号和填充符号）
        output_dim = 181  # 输出词汇表大小（需你确认）
        n_layers = 1
        OUTPUT_DIM = 181  # 输出词汇表大小（需你确认）
        max_length = 800/model_dict[model_name][2]

        # 参数列表
        kmeans_loaded = load(os.path.join(model_dict['root']['kmeans'],model_dict[model_name][1]))  # 分类文件夹根目录
        # save_path = '../SLR_dataset/kmeans_100_seq/'
        all_new_labels = []
        datalist = []
        for line in landmarks:  # 逐行读取文件内容
            # 按空格分割每一行，并将每个部分转换为浮点型
            # print(line) 每一行63*2个数据
            for hand in line:
                datalist.append(hand)  # 将转换后的浮点型列表添加到 datalist
        if '双手合并' in model_dict[model_name][1]:
            datalist = [datalist[2*i]+datalist[2*i+1] for i in range(len(datalist)//2)]
            max_length = int(max_length/2)
        new_labels = kmeans_loaded.predict(datalist)
        print(new_labels)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        max_length = 800  # 源序列最大长度
        end_token = (input_dim - 2)  # 源序列结束符号
        pad_token = (input_dim - 1)  # 源填充符号
        vocab = torch.load("../../model/vocab.pt", weights_only=False)
        model = torch.load(os.path.join(model_dict['root']['model'],model_dict[model_name][0]), weights_only=False)

        numpy_array = np.array(new_labels, dtype=np.int32)
        j_tensor = torch.tensor(numpy_array)
        # 1. 先添加结束符41（计入1500长度内）
        j_with_end = torch.cat([j_tensor, torch.tensor([end_token], dtype=j_tensor.dtype)])

        # 2. 处理长度
        if len(j_with_end) > max_length:
            # 如果超长：截断到1499再加结束符
            j_processed = torch.cat([j_with_end[:max_length - 1],
                                     torch.tensor([end_token], dtype=j_tensor.dtype)])
        elif len(j_with_end) < max_length:
            pad_needed = max_length - len(j_with_end)
            padding = torch.full((pad_needed,), pad_token, dtype=j_tensor.dtype)
            j_processed = torch.cat([j_with_end, padding])
        else:
            # 刚好1500
            j_processed = j_with_end

        # 验证长度
        assert len(j_processed) == max_length, f"长度错误：{len(j_processed)} != {max_length}"

        print(j_processed)
        j_processed = j_processed.unsqueeze(0)
        idx2word = vocab.get_itos()
        j_processed = j_processed.to(device)
        output = model.predict(j_processed)
        output = output[0].tolist()
        text = ''
        for i in output:
            text += idx2word[i]
        print(text)

        # Format the response
        response = {
            'status': 'success',
            'resulttext':text,
            'filename': filename,
            'filepath': save_path,
            'frame_count': len(landmarks),
            'new_labels': new_labels.tolist(),
            'landmarks': landmarks  # Return first 10 frames for demo
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        # Clean up if file was saved but processing failed
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)