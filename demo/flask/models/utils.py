ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

model_dict = {
    'root':{'model':'../../model','kmeans':'../../model/kmeans'},
    'Km20-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km20/lstm_kme20_emb256_hid256.pth',
        'kmeans_20.joblib',
        1,
        22
    ],
    'Km30-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km30/lstm_kme30_emb512_hid512.pth',
        'kmeans_30.joblib',
        1,
        32
    ],
    'Km40-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb128_hid128.pth',
        'kmeans_40.joblib',
        1,
        42
    ],
    'Km40-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb256_hid256.pth',
        'kmeans_40.joblib',
        1,
        42
    ],
    'Km40-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb512_hid512.pth',
        'kmeans_40.joblib',
        1,42
    ],
    'Km50-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km50/lstm_kme50_emb128_hid128.pth',
        'kmeans_50.joblib',
        1,52
    ],
    'Km50-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km50/lstm_kme50_emb256_hid256.pth',
        'kmeans_50.joblib',
        1,52
    ],
    'Km70-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km70/lstm_kme70_emb128_hid128.pth',
        'kmeans_70.joblib',
        1,72
    ],
    'Km70-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km70/lstm_kme70_emb256_hid256.pth',
        'kmeans_70.joblib',
        1,72
    ],
    'Km70-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km70/lstm_kme70_emb512_hid512.pth',
        'kmeans_70.joblib',
        1,72
    ],
    'Km100-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km100/lstm_kme100_emb128_hid128.pth',
        'kmeans_100.joblib',
        1,102
    ],
    'Km100-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km100/lstm_kme100_emb256_hid256.pth',
        'kmeans_100.joblib',
        1,102
    ],
    'Km100-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km100/lstm_kme100_emb512_hid512.pth',
        'kmeans_100.joblib',
        1,102
    ],
    'Km40-emb256-hid256_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'Km40-emb512-hid512_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb512_hid512_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'Km80-emb512-hid512_双手合并_抽帧1': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame1_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        1,82
    ],
    'Km80-emb512-hid512_双手合并_抽帧3': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame3_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        3,82
    ],
    'Km80-emb512-hid512_双手合并_抽帧5': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'Km80-emb512-hid512_双手合并_抽帧7': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame7_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        7,82
    ],
    'train0.8_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集八成_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.5_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集一半_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.3_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集三成_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.2_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集二成_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.8kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集八成_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.5kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集一半_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.3kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集三成_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.2kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集二成_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5,82
    ],
    'train0.8_kme40_emb256_hid256_multi-head_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb256_hid256_frame1_双手合并_多头qkv.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'train0.5_kme40_emb256_hid256_multi-head_双手合并_抽帧1': [
        'xuanmen_km40/训练集一半_lstm_kme40_emb256_hid256_frame1_双手合并_多头qkv.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'train0.3_kme40_emb256_hid256_multi-head_双手合并_抽帧1': [
        'xuanmen_km40/训练集三成_lstm_kme40_emb256_hid256_frame1_双手合并_多头qkv.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'train0.8_kme40_emb256_hid256_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'train0.5_kme40_emb256_hid256_双手合并_抽帧1': [
        'xuanmen_km40/训练集一半_lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'train0.3_kme40_emb256_hid256_双手合并_抽帧1': [
        'xuanmen_km40/训练集三成_lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1,42
    ],
    'train0.8_kme80_emb512_hid512_multi-head_双手合并_抽帧1': [
        'xuanmen_km80/训练集八成_lstm_kme80_emb512_hid512_frame1_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        1,82
    ],
    'train0.5_kme80_emb512_hid512_multi-head_双手合并_抽帧1': [
        'xuanmen_km80/训练集五成_lstm_kme80_emb512_hid512_frame1_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        1,82
    ],
    'train0.3_kme80_emb512_hid512_multi-head_双手合并_抽帧1': [
        'xuanmen_km80/训练集三成_lstm_kme80_emb512_hid512_frame1_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        1,82
    ]
}