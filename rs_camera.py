from pathlib import Path
import sys
import os



FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])

import numpy as np
import time
import cv2

import torch
import torch.backends.cudnn as cudnn

from tools import *
#from tools_backup import *

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from utils.augmentations import letterbox
import pyrealsense2 as rs


device = 0
weights = path + "/weights/yolov5s.pt"
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 1000
half=False
dnn = False

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size


half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

if pt:
    model.model.half() if half else model.model.float()

cudnn.benchmark = True  # set True to speed up constant image size inference


color = tuple(np.random.randint(low=200, high = 255, size = 3).tolist())
color = tuple([0,125,255])

recode = False
start_frame = 0
video_path = "test_video.mov"

# 2번 frame 1250



def person_tracking(model, img, img_ori, device):


        img_in = torch.from_numpy(img).to(device)
        img_in = img_in.float()
        img_in /= 255.0

        if img_in.ndimension() == 3:
            img_in = img_in.unsqueeze(0)
        

        pred = model(img_in, augment=False, visualize=False)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # detections per image
            
            im0 = img_ori.copy()

            if len(det):
                det[:, :4] = scale_coords(img_in.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class

                    label = names[c] #None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    x0, y0, x1, y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    x0, y0, x1, y1 = x0 - 10, y0 - 10, x1 + 10, y1

                    plot_one_box([x0, y0, x1, y1], im0, label=label, color=colors(c, True), line_thickness=3)
            
        return im0


def main(input_video):

    ball_esti_pos = []
    dT = 1 / 25

    pipeline=rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
    pipeline.start(config)




    if recode:
        codec = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter("ball_landing_point.mp4", codec, fps, (2144,810))


    disappear_cnt = 0
    ball_pos_jrajectory = []




    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            time.sleep(0.1)
            continue
        
        t1 = time.time()

        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.7), cv2.COLORMAP_JET)


        # frame = cv2.resize(frame, dsize = [0,0], fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LINEAR)


        img, img_ori = img_preprocessing(frame, imgsz, stride, pt)

        person_tracking_img = person_tracking(model, img.copy(), img_ori.copy(), device)

        #self.frame_recode = self.main_frame

        t2 = time.time()

        #main_frame = cv2.hconcat([self.camera_data, robot_detect_img, ball_image])

        cv2.imshow("person_tracking_img", person_tracking_img)
        cv2.imshow("depth_colormap", depth_colormap)
        #cv2.imshow("fgmask_dila", self.fgmask_dila)

        #cv2.imshow("ball_image", ball_image)

        print("FPS : ",1 / (t2 - t1))

        #print((t2-t1))
        key = cv2.waitKey(1)


        if key == 27 : 
            pipeline.stop()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":

    main(video_path)
