# Computer Vision Tasks

Five computer vision tasks using the programming language Python.

## Optical flow (Task 1)
Evaluating the implementation of two popular techniques for estimating optical flow from a sequence of images - the Lucas-Kanade and Horn-Schunck methods. Comparing their results by testing them on random noise and on pairs of images. Discovering the best parameters and techniques that help us improve the estimated optical flows and performances. Implementation and evaluation of the pyramidal Lucas-Kanade.

## Mean-Shift tracking (Task 2)
Implementation of the Mean-Shift mode seeking and its usage by the Mean-Shift tracker. Computing the convergence and computational efficiency by using different functions, starting points, kernel sizes/types, and termination criteria. Evaluation of the tracker on 5 different sequences from VOT14 with different parameters.

## Correlation filter tracking (Task 3)
Implementation of the MOSSE correlation filter tracker and of the actual MOSSE tracker. Comparing their tracking speed and performance by using different parameters, such as update factor, parameter Gaussian, and enlarge factor. The trackers are evaluated on all sequences from the dataset VOT14 as well as on each of the sequences.

## Advanced tracking (Task 4)
Implementation of three motion models (Random Walk, Nearly-Constant Velocity, Nearly-Constant Acceleration) using the Kalman filter. Evaluating each model with different values of the parameters q and r on three different curves. Proposing a particle filter tracker that uses NCV motion and a color histogram as a visual model. Evaluating the proposed tracker on the VOT14 sequence dataset using different parameters, motion models, number of particles, and colorspaces for the generation of the histogram.

## Long-term tracking (Task 5)
Implementation and evaluation of the short-term and long-term versions of the SiamFC tracker. Comparing and analyzing both performances in terms of Precision, Recall, and F-score metrics on a chosen long-term sequence called car9. Determination of optimal confidence threshold for initiating and terminating re-detection processes and investigating the impact of different numbers of randomly sampled regions during re-detection on the tracker’s ability to re-detect the target within fewer frames. Comparing different sampling strategies, such as random sampling and Gaussian sampling around the last confident position.
