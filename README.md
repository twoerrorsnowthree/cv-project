지윤 님은 best.pt 파일 사용하시면 됩니다!
정하 님은 보고서 작성하실 때, results.png, alpr_detection_yolo.py 사용하시면 됩니다!


#스니펫

from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
results = model.predict("차사진.jpg")

img = cv2.imread("차사진.jpg")
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    plate_crop = img[y1:y2, x1:x2]   # crop을 문자 인식 모델에 넣으면 됩니다!
