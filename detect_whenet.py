import cv2
import numpy as np
from PIL import Image

from utils import draw_axis
from whenet import WHENet
from yolo_v3.yolo_postprocess import YOLO

whenet = WHENet(snapshot='WHENet.h5')
yolo = YOLO()


def process_detection(model, img, bbox):
    y_min, x_min, y_max, x_max = bbox
    # enlarge the bbox to include more background margin
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, img.shape[1])

    img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), 2)
    yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

    return yaw, pitch, roll


def pred_frame(frame, **kwargs):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    bboxes, scores, classes = yolo.detect(img_pil)
    best_bbox = bboxes[np.argmax(scores)]
    yaw, pitch, roll = process_detection(whenet, frame, best_bbox)

    return {'pitch': pitch, 'yaw': yaw, 'roll': roll}


def draw_detection(img, bbox, yaw, pitch, roll):
    y_min, x_min, y_max, x_max = bbox
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, img.shape[1])

    draw_axis(img, yaw, pitch, roll, tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size=abs(x_max - x_min) // 2)

    cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (100, 255, 0), 1)
    cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (100, 255, 0), 1)
    cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (100, 255, 0), 1)
