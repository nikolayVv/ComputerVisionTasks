import os
import glob
import cv2
import numpy as np

from utils.utils import polygon2rectangle
from utils.io_utils import read_regions


class Sequence():

    def __init__(self, dataset_dir, sequence_name):

        self._name = sequence_name

        self.groundtruth = read_regions(os.path.join(dataset_dir, sequence_name, 'groundtruth.txt'))

        frames_dir = os.path.join(dataset_dir, sequence_name, 'color')
        if not os.path.exists(frames_dir):
            frames_dir = os.path.join(dataset_dir, sequence_name)

        self.frames = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))

        if len(self.frames) > len(self.groundtruth):
            self.frames = self.frames[len(self.frames) - len(self.groundtruth):]

    @property
    def length(self):
        return len(self.frames)

    @property
    def name(self):
        return self._name

    def read_frame(self, frame_index):
        return cv2.imread(self.frames[frame_index])

    def gt_region(self, frame_index, format='RECTANGLE'):
        
        region = self.groundtruth[frame_index]

        if format.upper() == 'RECTANGLE':

            if len(region) == 4:
                return region
            elif len(region) == 8:
                return polygon2rectangle(region)
            else:
                print('Unknown format of the groundtruth region:', region)
                exit(-1)

        elif format.upper() == 'POLYGON':

            if len(region) == 4:
                return rectangle2polygon(region)
            elif len(region) == 8:
                return region
            else:
                print('Unknown format of the groundtruth region:', region)
                exit(-1)
                
        else:
            print('Unknown output region format: %s. Supported only RECTANGLE and POLYGON.' % format)
            exit(-1)

        return self.groundtruth[frame_index]

    def visualize_results(self, regions, show_groundtruth=False):
        print('********************************************************')
        print('Use the following keys to control the video output:')
        print('* Space: Resume or pause video playback.')
        print('* D: Next frame (makes sense when the video is paused).')
        print('* A: Previous frame (makes sense when the video is paused).')
        print('* Esc: Quit video mode.')
        print('')
        print('Red rectangle: tracker\'s predicted region.')
        print('Green rectangle: ground-truth annotations.')
        print('********************************************************')

        win_name = 'Window'
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        wait_timeout = 0

        frame_idx = 0
        while frame_idx < self.length:

            img = self.read_frame(frame_idx)

            region = regions[frame_idx]

            if len(region) == 4:
                tl_ = (int(round(region[0])), int(round(region[1])))
                br_ = (int(round(region[0] + region[2] - 1)), int(round(region[1] + region[3] - 1)))
                cv2.rectangle(img, tl_, br_, (0, 0, 255), 2)
            elif len(region) == 8:
                p = np.round(np.array(region)).astype(np.int32)
                cv2.polylines(img, [p.reshape((-1, 1, 2))], True, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            if show_groundtruth:
                gt_reg = self.gt_region(frame_idx)
                tl_ = (int(round(gt_reg[0])), int(round(gt_reg[1])))
                br_ = (int(round(gt_reg[0] + gt_reg[2] - 1)), int(round(gt_reg[1] + gt_reg[3] - 1)))
                cv2.rectangle(img, tl_, br_, (0, 255, 0), 2)
            
            cv2.imshow(win_name, img)
            key_ = cv2.waitKey(wait_timeout)

            if wait_timeout > 0 and key_ == -1:
                frame_idx = min(frame_idx + 1, self.length - 1)

            if key_ == 27:
                break
            elif key_ == 32:
                if wait_timeout == 0:
                    wait_timeout = 30
                else:
                    wait_timeout = 0
            elif key_ == 100:
                frame_idx = min(frame_idx + 1, self.length - 1)
            elif key_ == 97:
                frame_idx = max(frame_idx - 1, 0)

        cv2.destroyWindow(win_name)
