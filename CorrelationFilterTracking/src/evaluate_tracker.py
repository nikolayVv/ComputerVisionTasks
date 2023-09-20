import argparse
import os

from utils.utils import load_tracker, load_dataset

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

def evaluate_tracker(workspace_path, tracker_id):
    tracker_class, tracker_name = load_tracker(workspace_path, tracker_id)
    if tracker_name == 'MosseTracker':
        tracker = tracker_class(enlarge_factor_mosse, alpha_mosse, beta_mosse, gamma_mosse, sigma_mosse, adaptive_scale_mosse)
    elif tracker_name == 'MosseTrackerImproved':
        tracker = tracker_class(enlarge_factor_mosse, alpha_mosse, beta_mosse, gamma_mosse, sigma_mosse, adaptive_scale_mosse, learning_rate, psr_threshold)
    elif tracker_name == 'MSTracker':
        tracker = tracker_class(enlarge_factor_ms, kernel_shape, n_bins, min_shifting, max_iter, adaptive_scale_ms, alpha_ms, gamma_ms, sigma_ms)
    elif tracker_name == 'NCCTracker':
        tracker = tracker_class()
    dataset = load_dataset(workspace_path)

    results_dir = os.path.join(workspace_path, 'results', tracker.name())
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    tracker.evaluate(dataset, results_dir)
    print('Evaluation has been completed successfully.')


def main():
    parser = argparse.ArgumentParser(description='Tracker Evaluation Utility')

    parser.add_argument('--workspace_path', help='Path to the VOT workspace', required=True, action='store')
    parser.add_argument('--tracker', help='Tracker identifier', required=True, action='store')

    args = parser.parse_args()

    evaluate_tracker(args.workspace_path, args.tracker)

if __name__ == "__main__":
    main()
