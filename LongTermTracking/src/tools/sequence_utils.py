import os
import glob
import math

import numpy as np
import cv2


class VOTSequence():

    def __init__(self, dataset_path, sequence_name):
        self.name = sequence_name
        self.sequence_path = os.path.join(dataset_path, sequence_name)
        self.frames = []
        self.gt = []
        self.window_name = ''
        self.load_sequence()

    def load_sequence(self):
        frames_path = os.path.join(self.sequence_path, 'color')
        if not os.path.exists(frames_path):
            frames_path = self.sequence_path

        self.frames = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
        self.gt = self.read_groundtruth(os.path.join(self.sequence_path, 'groundtruth.txt'))

    def frame(self, frame_idx):
        return self.frames[frame_idx]

    def length(self):
        return len(self.frames)

    def read_groundtruth(self, file_path):
        with open(file_path, 'r') as gt_file:
            gt_ = gt_file.readlines()
            return [[float(el) for el in line.strip().split(',')] for line in gt_]

    def get_annotation(self, frame_idx, type='rectangle'):
        if type == 'rectangle':
            return self.convert_region(self.gt[frame_idx], type)
        elif type == 'polygon':
            return self.convert_region(self.gt[frame_idx], type)
        else:
            print('Error: Unknown annotation format.')
            exit(-1)

    def visible_frames(self):
        visible = len(self.gt) * [float(1)]
        for i in range(len(self.gt)):
            if any([math.isnan(el) for el in self.gt[i]]):
                visible[i] = 0
        return visible

    def convert_region(self, region, type):
        if (len(region) == 4 and type == 'rectangle') or (len(region) == 8 and type == 'polygon'):
            return region
        elif len(region) == 8 and type == 'rectangle':
            #convert from polygon to rectangle using min-max rectangle
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            return [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        elif len(region) == 4 and type == 'polygon':
            x0 = region[0]
            y0 = region[1]
            x1 = x0 + region[2] - 1
            y1 = y0 + region[3] - 1
            return [x0, y0, x1, y0, x1, y1, x0, y1]
        else:
            print('Error: Cannot convert region.')
            exit(-1)

    def overlap(self, region1, region2):
        # simplified overlap: region1 and regions are converted into axis-aligned bounding boxes
        if any([math.isnan(el) for el in region1]) or any([math.isnan(el) for el in region2]):
            return 0

        bb1 = self.convert_region(region1, type='rectangle')
        bb2 = self.convert_region(region2, type='rectangle')
        # coordinates of the intersect
        xA = max(bb1[0], bb2[0])
        yA = max(bb1[1], bb2[1])
        xB = min(bb1[0] + bb1[2] - 1, bb2[0] + bb2[2] - 1)
        yB = min(bb1[1] + bb1[3] - 1, bb2[1] + bb2[3] - 1)
        # area of the intersect
        intersect_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # areas of both bboxes
        area1 = bb1[2] * bb1[3]
        area2 = bb2[2] * bb2[3]
        # IoU
        return intersect_area / float(area1 + area2 - intersect_area)

    # drawing functions
    def initialize_window(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def draw_region(self, img, region, color, line_width):
        if len(region) == 4:
            # rectangle
            tl = (int(round(region[0])), int(round(region[1])))
            br = (int(round(region[0] + region[2] - 1)), int(round(region[1] + region[3])))
            cv2.rectangle(img, tl, br, color, line_width)
        elif len(region) == 8:
            # polygon
            pts = np.round(np.array(region).reshape((-1, 1, 2))).astype(np.int32)
            cv2.polylines(img, [pts], True, color, thickness=line_width, lineType=cv2.LINE_AA)
        else:
            print('Error: Unknown region format.')
            exit(-1)

    def draw_text(self, img, text, text_pos):
        font = cv2.FONT_HERSHEY_PLAIN
        text_sz = cv2.getTextSize(text, font, 1, 1)
        tl_ = (text_pos[0] - 5, text_pos[1] + 5)
        br_ = (text_pos[0] - 5 + text_sz[0][0] + 10, text_pos[1] - 5 - text_sz[0][1])
        cv2.rectangle(img, tl_, br_, (0, 0, 0), cv2.FILLED)
        cv2.putText(img, text, text_pos, font, 1, (255, 255, 255), 1, cv2.LINE_AA, False)

    def show_image(self, img, delay):
        cv2.imshow(self.window_name, img)
        cv2.waitKey(delay)

def save_results(results, file_path):
    with open(file_path, 'w') as f:
        for result in results:
            if len(result) == 1:
                f.write(str(result[0]) + '\n')
            else:
                f.write(','.join([str(el) for el in result]) + '\n')

def read_results(file_path):
    with open(file_path, 'r') as f:
        results = f.readlines()
        return [[float(el) for el in line.strip().split(',')] for line in results]