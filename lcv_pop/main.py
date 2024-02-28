import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')

area1 = [(312,388),(289,390),(474,469),(497,462)]
area2 = [(279,392),(250,397),(423,477),(454,469)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('peoplecount1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0
tracker = Tracker()
people_entering = {}
entering = set()
exiting = set()
people_exiting = {}


while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    bbox_list = []
             
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            bbox_list.append([x1, y1, x2, y2])
            bbox_id = tracker.update(bbox_list)
            
            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                
                results_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
                results_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
                
                if results_area1 >= 0:
                    people_entering[id] = (x4, y4)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    
                if results_area2 >= 0:
                    people_exiting[id] = (x4, y4)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    
                if id in people_exiting and results_area2 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cv2.circle(frame, (x4, y4), 5, (255, 0, 255), 1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                    exiting.add(id)

                if id in people_entering and results_area1 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                    cv2.circle(frame, (x4, y4), 5, (255, 0, 0), 1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 255), 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    sum = exiting - entering
    print("จำนวนคนเข้า:", len(exiting))
    print("จำนวนคนออก:", len(entering))
    print("คนที่อยู่ในร้าน:", len(sum))

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
