ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_dict = {
    'root':['../../model','../../model/kmeans'],
    'kmeans,Km20-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km20/lstm_kme20_emb256_hid256.pth',
        'kmeans_20.joblib',
        1
    ],
    'kmeans,Km30-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km30/lstm_kme30_emb512_hid512.pth',
        'kmeans_30.joblib',
        1
    ],
    'kmeans,Km40-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km40/lstm_Km40-emb128-hid128.pth',
        'kmeans_40.joblib',
        1
    ],
    'kmeans,Km40-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km40/lstm_Km40-emb256-hid256.pth',
        'kmeans_40.joblib',
        1
    ],
    'kmeans,Km40-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km40/lstm_Km40-emb512-hid512.pth',
        'kmeans_40.joblib',
        1
    ],
    'kmeans,Km50-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km50/lstm_Km50-emb128-hid128.pth',
        'kmeans_50.joblib',
        1
    ],
    'kmeans,Km50-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km50/lstm_Km50-emb256-hid256.pth',
        'kmeans_50.joblib',
        1
    ],
    'kmeans,Km70-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km70/lstm_Km70-emb128-hid128.pth',
        'kmeans_70.joblib',
        1
    ],
    'kmeans,Km70-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km70/lstm_Km70-emb256-hid256.pth',
        'kmeans_70.joblib',
        1
    ],
    'kmeans,Km70-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km70/lstm_Km70-emb512-hid512.pth',
        'kmeans_70.joblib',
        1
    ],
    'kmeans,Km100-emb128-hid128_双手分离_抽帧1': [
        'xuanmen_km100/lstm_Km100-emb128-hid128.pth',
        'kmeans_100.joblib',
        1
    ],
    'kmeans,Km100-emb256-hid256_双手分离_抽帧1': [
        'xuanmen_km100/lstm_Km100-emb256-hid256.pth',
        'kmeans_100.joblib',
        1
    ],
    'kmeans,Km100-emb512-hid512_双手分离_抽帧1': [
        'xuanmen_km100/lstm_Km100-emb512-hid512.pth',
        'kmeans_100.joblib',
        1
    ],
    'kmeans,Km40-emb256-hid256_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1
    ],
    'kmeans,Km40-emb512-hid512_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb512_hid512_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1
    ],
    'kmeans,Km80-emb512-hid512_双手合并_抽帧1': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame1_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        1
    ],
    'kmeans,Km80-emb512-hid512_双手合并_抽帧3': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame3_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        3
    ],
    'kmeans,Km80-emb512-hid512_双手合并_抽帧5': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'kmeans,Km80-emb512-hid512_双手合并_抽帧7': [
        'xuanmen_km80/lstm_kme80_emb512_hid512_frame7_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        7
    ],
    'train0.8_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集八成_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.5_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集一半_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.3_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集三成_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.2_kme80_emb512_hid512_双手合并_抽帧5': [
        'xuanmen_km80/训练集二成_lstm_kme80_emb512_hid512_frame5_双手合并.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.8kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集八成_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.5kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集一半_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.3kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集三成_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.2kme80_emb512_hid512_multi-head_双手合并_抽帧5': [
        'xuanmen_km80/训练集二成_lstm_kme80_emb512_hid512_frame5_双手合并_多头qkv.pth',
        'kmeans_80_双手合并.joblib',
        5
    ],
    'train0.8_kme40_emb256_hid256_multi-head_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb256_hid256_frame1_双手合并_多头qkv.pth',
        'kmeans_40_双手合并.joblib',
        1
    ],
    'train0.5_kme40_emb256_hid256_multi-head_双手合并_抽帧1': [
        'xuanmen_km40/训练集一半_lstm_kme40_emb256_hid256_frame1_双手合并_多头qkv.pth',
        'kmeans_40_双手合并.joblib',
        1
    ],
    'train0.3_kme40_emb256_hid256_multi-head_双手合并_抽帧1': [
        'xuanmen_km40/训练集三成_lstm_kme40_emb256_hid256_frame1_双手合并_多头qkv.pth',
        'kmeans_40_双手合并.joblib',
        1
    ],
    'train0.8_kme40_emb256_hid256_双手合并_抽帧1': [
        'xuanmen_km40/lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1
    ],
    'train0.5_kme40_emb256_hid256_双手合并_抽帧1': [
        'xuanmen_km40/训练集一半_lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1
    ],
    'train0.3_kme40_emb256_hid256_双手合并_抽帧1': [
        'xuanmen_km40/训练集三成_lstm_kme40_emb256_hid256_frame1_双手合并.pth',
        'kmeans_40_双手合并.joblib',
        1
    ]
}