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

app = Flask(__name__)
CORS(app)
# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class seq2seq(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.encode_embedding = nn.Embedding(input_dim, emb_dim)  # 将每个词扩充为emb_dim维
        self.decode_embedding = nn.Embedding(output_dim, emb_dim)
        self.encode = nn.LSTM(emb_dim, hidden_dim, n_layers)
        self.decode = nn.LSTM(emb_dim, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tar):
        # src: [batch_size, src_len]
        # tar: [batch_size, trg_len]

        # 编码器部分
        encode_embedded = self.encode_embedding(src)  # [batch_size, src_len, emb_dim]
        encode_embedded = encode_embedded.permute(1, 0, 2)  # [src_len, batch_size, emb_dim]
        _, (hidden, cell) = self.encode(encode_embedded)

        # 解码器部分
        batch_size = tar.shape[0]  # 3
        trg_len = tar.shape[1]  # 9
        output_dim = self.fc.out_features  # 181
        # print(output_dim)

        # 准备输出张量
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(src.device)  # 9x3x181

        # 初始输入是<sos> token，这里假设tar已经包含<sos>作为第一个token
        input = tar[:, 0]  # 取第一个token作为初始输入 [batch_size]

        for t in range(1, trg_len):
            # 嵌入输入
            embedded = self.decode_embedding(input).unsqueeze(0)  # [1, batch_size, emb_dim]
            # print(f'embedded:{embedded.size()}')
            # print(f'hidden:{hidden.size()}')
            # 通过解码器
            output, (hidden, cell) = self.decode(embedded, (hidden, cell))

            # 预测下一个token
            pred = self.fc(output.squeeze(0))
            outputs[t] = pred

            # 下一个输入是真实目标(teacher forcing)或预测结果
            # 这里使用teacher forcing，传入真实目标
            input = tar[:, t]

        return outputs.permute(1, 0, 2)  # [batch_size, trg_len, output_dim]

    def predict(self, src, sos_token_idx=0, eos_token_idx=1, max_len=9):
        """
        自回归预测（不需要输入tar）
        :param src: 输入序列 [batch_size, src_len]
        :param sos_token_idx: <sos>的索引
        :param eos_token_idx: <eos>的索引（可选）
        :param max_len: 最大生成长度
        :return: 预测序列 [batch_size, max_len]
        """
        # 编码器部分
        encode_embedded = self.encode_embedding(src).permute(1, 0, 2)
        _, (hidden, cell) = self.encode(encode_embedded)

        # 解码器初始化
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, max_len).long().to(src.device)
        input = torch.full((batch_size,), sos_token_idx, dtype=torch.long).to(src.device)

        # 自回归解码
        for t in range(max_len):
            embedded = self.decode_embedding(input).unsqueeze(0)  # [1, batch_size, emb_dim]
            output, (hidden, cell) = self.decode(embedded, (hidden, cell))
            pred = self.fc(output.squeeze(0)).argmax(-1)  # [batch_size]

            outputs[:, t] = pred
            input = pred  # 使用预测结果作为下一输入

            # 如果所有序列都生成<eos>则提前停止
            if eos_token_idx is not None and (pred == eos_token_idx).all():
                break

        return outputs

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_with_mediapipe(video_path):
    """Process video with MediaPipe and return hand landmarks"""
    print("Starting video processing...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.3
    )

    landmarks_data = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get wrist (root) coordinates
                    root = hand_landmarks.landmark[0]
                    hand_data = []

                    for landmark in hand_landmarks.landmark:
                        # Calculate relative coordinates
                        relative_x = landmark.x - root.x
                        relative_y = landmark.y - root.y
                        relative_z = landmark.z - root.z
                        hand_data.extend([relative_x, relative_y, relative_z])

                    frame_landmarks.append(hand_data)

            # Pad with zeros if less than 2 hands detected
            while len(frame_landmarks) < 2:
                frame_landmarks.append([0.0] * 21 * 3)

            landmarks_data.append(frame_landmarks)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        print(f"Finished processing. Total frames: {frame_count}")
        return landmarks_data

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        hands.close()
        print("Released resources")


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
        landmarks = process_video_with_mediapipe(save_path)
        # print(landmarks)


        # 参数列表
        kmeans_loaded = load('../../model/kmeans/kmeans_40.joblib')  # 分类文件夹根目录
        all_path = '../SLR_dataset/seq_txt/'
        save_path = '../SLR_dataset/kmeans_100_seq/'

        def left_right_hand(data):
            left = data[0:63]
            right = data[63:]
            return left, right

        def list_proLR(data):
            # 该函数用于将读取到的数据变成左右手格式
            result = []
            for i in data:
                left, right = left_right_hand(i)
                result += [left, right]
            return result

        all_new_labels = []
        datalist = []

        datalist = []
        for line in landmarks:  # 逐行读取文件内容
            # 按空格分割每一行，并将每个部分转换为浮点型
            # print(line) 每一行63*2个数据
            for hand in line:
                datalist.append(hand)  # 将转换后的浮点型列表添加到 datalist

        new_labels = kmeans_loaded.predict(datalist)
        print(new_labels)



        # 模型参数
        input_dim = 42  # 输入词汇表大小(等于原词汇表大小+2，+2加的是结束符号和填充符号）
        emb_dim = 256  # 词向量维度
        hidden_dim = 256  # LSTM隐藏层维度
        output_dim = 181  # 输出词汇表大小（需你确认）
        n_layers = 1
        OUTPUT_DIM = 181  # 输出词汇表大小（需你确认）

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        max_length = 800  # 源序列最大长度
        end_token = (input_dim - 2)  # 源序列结束符号
        pad_token = (input_dim - 1)  # 源填充符号
        vocab = torch.load("../../model/vocab.pt", weights_only=False)
        model = torch.load('../../model/xuanmen_km40/lstm_kme40_emb256_hid256.pth', weights_only=False)

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