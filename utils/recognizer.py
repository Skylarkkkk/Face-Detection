import cv2
from .tools import load_names, get_latest_model_path
import numpy as np

def recognize_from_image(img_path):
    print("Press 'q' to quit the image window.")
    model_path = get_latest_model_path()
    if not model_path:
        print("No model found.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    names = load_names()
    # 用 fromfile 读取字节流（兼容中文路径）
    img_array = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"图像解码失败: {img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        label = names[id_] if id_ < len(names) else "Unknown"
        color = (0, 255, 0) if confidence < 100 else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{label} ({(100 - confidence):.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Image Result", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        print("Exiting image mode.")
        cv2.destroyAllWindows()

def recognize_from_camera():
    print("Press 'q' to quit the camera window.")
    model_path = get_latest_model_path()
    if not model_path:
        print("No model found.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    names = load_names()
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            label = names[id_] if id_ < len(names) else "Unknown"
            color = (0, 255, 0) if confidence < 100 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({(100 - confidence):.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting camera mode.")
            break
    cap.release()
    cv2.destroyAllWindows()
