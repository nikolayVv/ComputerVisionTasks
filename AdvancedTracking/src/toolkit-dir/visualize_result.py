import argparse
import os

from utils.utils import load_tracker, load_dataset 
from utils.io_utils import read_regions

# PARTICLE FILTER parameters
enlarge_factor_pf = 1
num_particles = 500
bins = 12
hist_color = 'HSV'
motion_model = 'NCV'
q_factor = 0.1
alpha_pf = 0.05
distance_sigma_pf = 0.1
kernel_sigma_pf = 0.5

def visualize_tracking_result(workspace_path, tracker_id, sequence_name, show_gt):

    dataset = load_dataset(workspace_path)

    sequence = None
    for sequence_ in dataset.sequences:
        if sequence_.name == sequence_name:
            sequence = sequence_
            break

    if sequence is None:
        print('Sequence (%s) cannot be found.' % sequence_name)
        exit(-1)

    tracker_class, _ = load_tracker(workspace_path, tracker_id)
    tracker = tracker_class(enlarge_factor_pf, num_particles, bins, hist_color, motion_model, q_factor, alpha_pf, distance_sigma_pf, kernel_sigma_pf)
        
    results_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name, '%s_%03d.txt' % (sequence.name, 1))
    if not os.path.exists(results_path):
        print('Results does not exist (%s).' % results_path)

    regions = read_regions(results_path)

    sequence.visualize_results(regions, show_groundtruth=show_gt)

def main():
    parser = argparse.ArgumentParser(description='Tracking Visualization Utility')

    parser.add_argument('--workspace_path', help='Path to the VOT workspace', required=True, action='store')
    parser.add_argument('--tracker', help='Tracker identifier', required=True, action='store')
    parser.add_argument('--sequence', help='Sequence name', required=True, action='store')
    parser.add_argument('--show_gt', help='Show groundtruth annotations', required=False, action='store_true')

    args = parser.parse_args()

    visualize_tracking_result(args.workspace_path, args.tracker, args.sequence, args.show_gt)

if __name__ == "__main__":
    main()
