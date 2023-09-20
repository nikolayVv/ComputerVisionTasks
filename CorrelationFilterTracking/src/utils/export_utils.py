import os
import json
import math

import matplotlib.pyplot as plt

from utils.dataset import Dataset
from utils.tracker import Tracker
from utils.plot_styles import load_plot_styles


def print_summary(output_dict):
    print('------------------------------------')
    print('Results for tracker:', output_dict['tracker_name'])
    print('  Average overlap: %.2f' % output_dict['average_overlap'])
    print('  Total failures: %.1f' % output_dict['total_failures'])
    print('  Average speed: %.2f FPS' % output_dict['average_speed'])
    print('  Average initialization speed: %.2f FPS' % output_dict['average_initialization_speed'])
    print('------------------------------------')

def load_output(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def export_plot(outputs, sensitivity, output_path):

    styles = load_plot_styles()

    if len(outputs) > len(styles):
        print('Number of compared trackers is larger than number of plot stlyes.')
        print('Modify the script utils/plot_styles.py by adding more plot styles and re-run.')
        exit(-1)
    
    fig = plt.figure()
    for output, style in zip(outputs, styles):
        a = output['average_overlap']
        r = math.exp(- sensitivity * (float(output['total_failures']) / float(output['total_frames'])))
        plt.plot(r, a, marker=style['marker'], markerfacecolor=style['color'], markeredgewidth=0, linestyle='', markersize=10, label=output['tracker_name'])    
    
    plt.axis('square')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.title('AR Plot')
    fig.axes[0].set_xlabel('Robustness')
    fig.axes[0].set_ylabel('Accuracy')
    fig.savefig(output_path)

    print('The AR plot is saved to the file:', output_path)

def export_measures(workspace_path: str, dataset: Dataset, tracker: Tracker, overlaps: list, failures: list, times: list, init_times: list):
    
    # create per-sequence output structure
    speed = len(dataset.sequences) * [0]
    init_speed = len(dataset.sequences) * [0]
    results = len(dataset.sequences) * [0]
    for i, sequence in enumerate(dataset.sequences):
        speed_fps = 1.0 / times[i]
        init_speed_fps = 1.0 / init_times[i]
        results[i] = {'sequence_name': sequence.name, 'sequence_length': sequence.length, \
            'overlap': overlaps[i], 'failures': failures[i], 'speed': speed_fps, 'initialization_speed': init_speed_fps}
        speed[i] = speed_fps
        init_speed[i] = init_speed_fps

    # average measures
    average_overlap = sum(overlaps) / len(dataset.sequences)
    total_failures = sum(failures)
    average_speed = sum(speed) / len(dataset.sequences)
    average_init_speed = sum(init_speed) / len(dataset.sequences)

    # final output structure with all information
    output = {'tracker_name': tracker.name(), 'results': results, 'average_overlap': average_overlap, \
        'total_failures': total_failures, 'average_speed': average_speed, 'average_initialization_speed': average_init_speed, 'total_frames': dataset.number_frames}

    # create output directory and save output in json file
    output_dir = os.path.join(workspace_path, 'analysis', tracker.name())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'results.json')

    with open(file_path, 'w') as f:
        json.dump(output, f, indent=2)

    print_summary(output)

    return output
