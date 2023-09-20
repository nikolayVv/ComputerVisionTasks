import argparse
import os
import math
import cv2
import numpy as np

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import read_results
import matplotlib.pyplot as plt


def estimate_thresholds(scores, n):
    
    scores_vector = np.sort(np.concatenate([np.array(s_) for s_ in scores], axis=0), axis=0)
    step = int(math.floor(scores_vector.shape[0] / (n - 1)))

    thresholds = [scores_vector[(i + 1) * step][0] for i in range(n - 2)]
    thresholds.insert(0, float('-inf'))
    thresholds.insert(len(thresholds), float('inf'))
    
    return thresholds[::-1]

def calculate_pr_re_f(sequences, overlaps, scores, thresholds):

    pr = len(thresholds) * [float(0)]
    re = len(thresholds) * [float(0)]

    for i, sequence in enumerate(sequences):
        
        seq_overlaps = np.array(overlaps[i]).flatten()
        seq_scores = np.array(scores[i]).flatten()
        visible = np.array(sequence.visible_frames()).flatten()

        n_visible = np.sum(visible)
        
        seq_pr = len(thresholds) * [float(0)]
        seq_re = len(thresholds) * [float(0)]

        for ti, thr in enumerate(thresholds):

            selector = seq_scores > thr
            
            if not any(selector):
                seq_pr[ti] = 1
                seq_re[ti] = 0
            else:
                seq_pr[ti] = np.mean(seq_overlaps[selector])
                seq_re[ti] = np.sum(seq_overlaps[selector]) / n_visible

        pr = [seq_pr_ + pr_ for seq_pr_, pr_ in zip(seq_pr, pr)]
        re = [seq_re_ + re_ for seq_re_, re_ in zip(seq_re, re)]

    pr = [pr_ / len(sequences) for pr_ in pr]
    re = [re_ / len(sequences) for re_ in re]
    f = [(2 * pr_ * re_) / (pr_ + re_) for pr_, re_ in zip(pr, re)]

    max_index = np.argmax(np.array(f))
    
    pr_score = pr[max_index]
    re_score = re[max_index]
    f_score = f[max_index]

    return pr_score, re_score, f_score

def evaluate_performance(dataset_path, results_dir):
    
    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    scores_all = []
    overlaps_all = []

    dataset = []
    for sequence_name in sequences:
        
        sequence = VOTSequence(dataset_path, sequence_name)
        dataset.append(sequence)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        bboxes = read_results(bboxes_path)
        scores = read_results(scores_path)

        if len(sequence.gt) != len(bboxes):
            print('Groundtruth and results does not have the same number of elements.')
            exit(-1)

        overlaps = [sequence.overlap(bb, gt) for bb, gt in zip(bboxes, sequence.gt)]
        
        scores_all.append(scores)
        overlaps_all.append(overlaps)

        threshold = estimate_thresholds([scores], 100)

        pr, re, f = calculate_pr_re_f([sequence], [overlaps], [scores], threshold)

        print(f'------------{sequence_name} START--------------')
        print('Precision:', pr)
        print('Recall:', re)
        print('F-score:', f)
        print(f'-------------{sequence_name} END-------------')

    thresholds = estimate_thresholds(scores_all, 100)
    pr, re, f = calculate_pr_re_f(dataset, overlaps_all, scores_all, thresholds)

    print('--------------------------')
    print('Precision:', pr)
    print('Recall:', re)
    print('F-score:', f)
    print('--------------------------')

parser = argparse.ArgumentParser(description='Longe-term Tracking Performance Evaluation Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')

args = parser.parse_args()

evaluate_performance(args.dataset, args.results_dir)
