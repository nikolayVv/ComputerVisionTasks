import argparse
import os

from utils.utils import load_tracker, load_dataset


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

def evaluate_tracker(workspace_path, tracker_id):
    tracker_class, _ = load_tracker(workspace_path, tracker_id)

    tracker = tracker_class(enlarge_factor_pf, num_particles, bins, hist_color, motion_model, q_factor, alpha_pf, distance_sigma_pf, kernel_sigma_pf)
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
