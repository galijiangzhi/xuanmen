import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 读取并显示视频流
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果读取失败，退出循环
    if not ret:
        print("无法读取帧")
        break

    # 显示帧
    cv2.imshow('Camera Feed', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()