import argparse
import os

from utils.utils import load_tracker, load_dataset
from utils.export_utils import load_output, print_summary, export_plot
from calculate_measures import tracking_analysis


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

def tracking_comparison(workspace_path, tracker_ids, sensitivity, output_path):

    dataset = load_dataset(workspace_path)
    outputs_all = []
    for tracker_id in tracker_ids:
        tracker_class, _ = load_tracker(workspace_path, tracker_id)
        tracker = tracker_class(enlarge_factor_pf, num_particles, bins, hist_color, motion_model, q_factor, alpha_pf, distance_sigma_pf, kernel_sigma_pf)
            
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