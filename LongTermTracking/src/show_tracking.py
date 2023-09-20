import argparse
import os
import math
import cv2
import numpy as np

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import read_results


def visualize_results(dataset_path, results_dir, sequence_name):
    
    sequence = VOTSequence(dataset_path, sequence_name)

    bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
    scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

    bboxes = read_results(bboxes_path)
    scores = read_results(scores_path)

    if len(sequence.gt) != len(bboxes):
        print('Groundtruth and results does not have the same number of elements.')
        exit(-1)

    overlaps = [sequence.overlap(bb, gt) for bb, gt in zip(bboxes, sequence.gt)]
        
    sequence.initialize_window('Window')

    for i in range(sequence.length()):

        img = cv2.imread(sequence.frame(i))

        gt_ = sequence.get_annotation(i)
        if not any([math.isnan(el) for el in gt_]):
            sequence.draw_region(img, gt_, (0, 255, 0), 2)
        sequence.draw_region(img, bboxes[i], (0, 0, 255), 2)

        sequence.draw_text(img, '%d/%d' % (i + 1, sequence.length()), (50, 25))
        sequence.draw_text(img, 'Score: %.3f' % scores[i][0], (50, 50))
        sequence.draw_text(img, 'Overlap: %.2f' % overlaps[i], (50, 75))

        sequence.show_image(img, 10)


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
parser.add_argument("--sequence", help="Sequence to visualize", required=True, action='store')

args = parser.parse_args()

visualize_results(args.dataset, args.results_dir, args.sequence)