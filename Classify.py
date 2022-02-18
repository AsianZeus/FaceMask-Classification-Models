from PIL import Image
import cv2
from RetinaFace_Detector import RetinaFace

def classify(image, model):
    count= 0
    labels = []
    obj = RetinaFace.detect_faces(image)
    for key in obj:
        count+=1
        identity = obj[key]
        facial_area = identity["facial_area"]
        face = image[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        (startX, startY, endX, endY) = (facial_area[0], facial_area[1], facial_area[2], facial_area[3])
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        res = model(face)
        label = res[0]['label'] if res[0]['score']> res[1]['score'] else res[1]['label']
        labels.append(label)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(image, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return (count, labels, image)
