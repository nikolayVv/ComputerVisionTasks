import argparse
import os

from utils.utils import load_tracker, load_dataset, trajectory_overlaps, count_failures, average_time
from utils.io_utils import read_regions, read_vector
from utils.export_utils import export_measures

# MOSSE PARAMETERS
# Enlarge factor -> 1, 1.25, 1.5, 2, 3, 4
enlarge_factor_mosse = 1.5
adaptive_scale_mosse = 1e-8
# Update filter factor -> 0.02, 0.05, 0.1, 0.2, 0.3, 0.5
alpha_mosse = 0.3
# Update adaptive scale factor
beta_mosse = 0.1
# Denominator factor
gamma_mosse = 0.1
# Parameter for Gaussian -> 0.5, 1, 2, 3, 4
sigma_mosse = 2
# Update numerator and denominator factor
learning_rate = 0.75
psr_threshold = 7

# MS parameters
enlarge_factor_ms = 1
kernel_shape = 'epanechnikov'
n_bins = 12
min_shifting = 1
max_iter = 20
adaptive_scale_ms = 1e-8
alpha_ms = 1e-8
gamma_ms = 0.1
sigma_ms = 1

def tracking_analysis(workspace_path, tracker_id):

    dataset = load_dataset(workspace_path)

    tracker_class, tracker_name = load_tracker(workspace_path, tracker_id)
    if tracker_name == 'MosseTracker':
        tracker = tracker_class(enlarge_factor_mosse, alpha_mosse, beta_mosse, gamma_mosse, sigma_mosse, adaptive_scale_mosse)
    elif tracker_name == 'MosseTrackerImproved':
        tracker = tracker_class(enlarge_factor_mosse, alpha_mosse, beta_mosse, gamma_mosse, sigma_mosse, adaptive_scale_mosse, learning_rate, psr_threshold)
    elif tracker_name == 'MSTracker':
        tracker = tracker_class(enlarge_factor_ms, kernel_shape, n_bins, min_shifting, max_iter, adaptive_scale_ms, alpha_ms, gamma_ms, sigma_ms)
    elif tracker_name == 'NCCTracker':
        tracker = tracker_class()

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
