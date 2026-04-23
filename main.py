# =============================
# FACE DETECTION USING OPENCV
# =============================

import cv2

# Load model nhận diện mặt (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit")

while True:
    # Đọc frame
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Vẽ khung
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Hiển thị
    cv2.imshow('Face Detection', frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng
cap.release()
cv2.destroyAllWindows()