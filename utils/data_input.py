import cv2
import os

def capture_images(name, save_dir="./data"):
    print("Press 's' to save an image, 'q' to quit.")
    cap = cv2.VideoCapture(0)
    num = 0
    os.makedirs(save_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 检测是否已经存在同名文件
            while os.path.exists(os.path.join(save_dir, f"{name}{num}.jpg")):
                num += 1
            # 保存图像
            path = os.path.join(save_dir, f"{name}{num}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved: {path}")
            num += 1
        elif key == ord('q'):
            print("Exiting capture mode.")
            break

    cap.release()
    cv2.destroyAllWindows()
