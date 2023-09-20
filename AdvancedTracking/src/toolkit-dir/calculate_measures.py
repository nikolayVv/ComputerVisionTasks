import argparse
import os

from utils.utils import load_tracker, load_dataset, trajectory_overlaps, count_failures, average_time
from utils.io_utils import read_regions, read_vector
from utils.export_utils import export_measures


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

def tracking_analysis(workspace_path, tracker_id):

    dataset = load_dataset(workspace_path)

    tracker_class, _ = load_tracker(workspace_path, tracker_id)
    tracker = tracker_class(enlarge_factor_pf, num_particles, bins, hist_color, motion_model, q_factor, alpha_pf, distance_sigma_pf, kernel_sigma_pf)
    print('Performing evaluation for tracker:', tracker.name())

    per_seq_overlaps = len(dataset.sequences) * [0]
    per_seq_failures = len(dataset.sequences) * [0]
    per_seq_time = len(dataset.sequences) * [0]
    per_seq_init_time = len(dataset.sequences) * [0]

    for i, sequence in enumerate(dataset.sequences):
        
        results_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name, '%s_%03d.txt' % (sequence.name, 1))
        if not os.path.exists(results_path):
            print('Results does not exist (%s).' % results_path)
        
        time_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name, '%s_%03d_time.txt' % (sequence.name, 1))
        if not os.path.exists(time_path):
            print('Time file does not exist (%s).' % time_path)

        init_time_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name, '%s_%03d_init_time.txt' % (sequence.name, 1))
        if not os.path.exists(init_time_path):
            print('Initialization time file does not exist (%s).' % time_path)

        regions = read_regions(results_path)
        times = read_vector(time_path)
        init_times = read_vector(init_time_path)

        overlaps, overlap_valid = trajectory_overlaps(regions, sequence.groundtruth)
        failures = count_failures(regions)
        t = average_time(times, regions)
        if failures != 0:
            init_t = sum(init_times) / failures
        else:
            init_t = 0

        per_seq_overlaps[i] = sum(overlaps) / sum(overlap_valid)
        per_seq_failures[i] = failures
        per_seq_time[i] = t
        per_seq_init_time[i] = init_t
    
    return export_measures(workspace_path, dataset, tracker, per_seq_overlaps, per_seq_failures, per_seq_time, per_seq_init_time)


def main():
    parser = argparse.ArgumentParser(description='Tracking Visualization Utility')

    parser.add_argument('--workspace_path', help='Path to the VOT workspace', required=True, action='store')
    parser.add_argument('--tracker', help='Tracker identifier', required=True, action='store')

    args = parser.parse_args()

    tracking_analysis(args.workspace_path, args.tracker)

if __name__ == "__main__":
    main()
