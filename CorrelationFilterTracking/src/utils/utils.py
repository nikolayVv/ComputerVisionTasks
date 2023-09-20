import os
import sys
import importlib.util
import yaml


def polygon2rectangle(p):
    x0 = min(p[::2])
    y0 = min(p[1::2])
    x1 = max(p[::2])
    y1 = max(p[1::2])
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

def rectangle2polygon(r):
    return [r[0], r[1], r[0] + r[2], r[1], r[0] + r[2], r[1] + r[3], r[0], r[1] + r[3]]

def calculate_overlap(a: list, b: list):
    if len(a) == 8:
        a = polygon2rectangle(a)
    if len(b) == 8:
        b = polygon2rectangle(b)

    if len(a) != 4 or len(b) != 4:
        print('Both regions must have 4 elements (bounding box) to calcualte overlap.')
        exit(-1)

    if a[2] < 1 or a[3] < 1 or b[2] < 1 or b[3] < 1:
        return 0

    intersection_area = max(0, min(a[0] + a[2] - 1, b[0] + b[2] - 1) - max(a[0], b[0])) * max(0, min(a[1] + a[3] - 1, b[1] + b[3] - 1) - max(a[1], b[1]))
    union_area = a[2] * a[3] + b[2] * b[3] - intersection_area

    return intersection_area / union_area

def trajectory_overlaps(t1: list, t2: list):
    # calcualte per-frame overlap for a trajectory (multiple frames)
    if len(t1) != len(t2):
        print('Error: Trajectories must be the same length.')
        print(-1)
    
    overlaps = len(t1) * [0]
    valid = len(t1) * [0]
    for i, (reg1, reg2) in enumerate(zip(t1, t2)):
        if len(reg1) > 1 and len(reg2) > 1:
            overlaps[i] = calculate_overlap(reg1, reg2)
            valid[i] = 1
    
    return overlaps, valid

def count_failures(regions):
    failures = 0
    for region in regions:
        if len(region) == 1 and region[0] == 2:
            failures += 1
    return failures

def average_time(times, regions):
    if len(times) != len(regions):
        print('Error: Number of time measurements must be the same as regions.')
        exit(-1)
    
    time_sum = float(0)
    valid_frames = 0
    for t, reg in zip(times, regions):
        if (len(reg) > 1) or (len(reg) == 1 and reg[0] > 0):
            time_sum += t
            valid_frames += 1
    
    return time_sum / valid_frames

from utils.tracker import Tracker
def load_tracker(workspace_path, tracker_id):

    tracker_config = None
    with open(os.path.join(workspace_path, 'trackers.yaml'), 'r') as yfile:
        trackers = yaml.load(yfile, Loader=yaml.BaseLoader)
        if tracker_id in trackers:
            tracker_config = trackers[tracker_id]
        else:
            print('Tracker %s is not in the trackers.yaml file.' % tracker_id)
            exit(-1)
    
    tracker_path = tracker_config['tracker_path']
    tracker_class_name = tracker_config['class_name']

    if not os.path.isfile(tracker_path):
        print('The variable tracker_path is not a file.')
        exit(-1)
    
    # det directory od the tracker file and add it to PYTHONPATH
    sys.path.insert(0, os.path.dirname(tracker_path))

    # check if another paths are specified and add them
    if 'paths' in tracker_config:
        for p in tracker_config['paths']:
            if p not in sys.path:
                sys.path.insert(0, p)
    
    spec = importlib.util.spec_from_file_location(tracker_class_name, tracker_path)
    module_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_)
    tracker_class = getattr(module_, tracker_class_name)
    return tracker_class, tracker_class_name

from utils.dataset import Dataset
def load_dataset(workspace_path):
    return Dataset(workspace_path)
    