import argparse
import os

from utils.utils import load_tracker, load_dataset
from utils.export_utils import load_output, print_summary, export_plot
from calculate_measures import tracking_analysis

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

def tracking_comparison(workspace_path, tracker_ids, sensitivity, output_path):

    dataset = load_dataset(workspace_path)
    outputs_all = []
    for tracker_id in tracker_ids:
        tracker_class, tracker_name = load_tracker(workspace_path, tracker_id)
        if tracker_name == 'MosseTracker':
            tracker = tracker_class(enlarge_factor_mosse, alpha_mosse, beta_mosse, gamma_mosse, sigma_mosse, adaptive_scale_mosse)
        elif tracker_name == 'MosseTrackerImproved':
            tracker = tracker_class(enlarge_factor_mosse, alpha_mosse, beta_mosse, gamma_mosse, sigma_mosse, adaptive_scale_mosse, learning_rate, psr_threshold)
        elif tracker_name == 'MSTracker':
            tracker = tracker_class(enlarge_factor_ms, kernel_shape, n_bins, min_shifting, max_iter, adaptive_scale_ms, alpha_ms, gamma_ms, sigma_ms)
        elif tracker_name == 'NCCTracker':
            tracker = tracker_class()
        results_path = os.path.join(workspace_path, 'analysis', tracker.name(), 'results.json')
        if os.path.exists(results_path):
            output = load_output(results_path)
            print_summary(output)
        else:
            output = tracking_analysis(workspace_path, tracker_id)
        
        outputs_all.append(output)

    if output_path == '':
        output_path = os.path.join(workspace_path, 'analysis', 'ar.png')

    export_plot(outputs_all, sensitivity, output_path)


def main():
    parser = argparse.ArgumentParser(description='Tracking Visualization Utility')

    parser.add_argument('--workspace_path', help='Path to the VOT workspace', required=True, action='store')
    parser.add_argument('--trackers', help='Tracker identifiers', required=True, action='store', nargs='*')
    parser.add_argument('--sensitivity', help='Sensitivtiy parameter for robustness', default=100, type=int)
    parser.add_argument('--output_path', help='Path for the output image', default='', type=str)

    args = parser.parse_args()
    
    tracking_comparison(args.workspace_path, args.trackers, args.sensitivity, args.output_path)

if __name__ == "__main__":
    main()