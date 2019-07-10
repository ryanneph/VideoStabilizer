# Offline Optimization-based Video Stabilizer
Here I have implemented an offline video stabilizer that places customizable constraints on maximum camera translation, rotation, and zoom, and enforces smooth motion by regularizing the first and second order finite differences (velocity, acceleration) of the camera motion parameters, while optimizing the smoothed camera motion to emulate the original camera motion (intentional camera movements).

A convex objective is formulated and solved using the FISTA (fast iterative shrinkage and thresholding) algorithm which has been successfully used to solve many large-scale convex optimization problems.

OpenCV is used to extract image features for each video frame, establish a robust correspondence between the features of adjacent frames, and estimate the affine camera motion parameters (translation, rotation, and scale/zoom). Once motion parameters are extracted for every frame, the motion sequence is optimized using FISTA to produce the smoothed camera motion, while still capturing the original intent of the video. Finally, the smoothed camera motion is applied to the video sequence to obtain a stabilized result, free from annoying shakes and jitters.

Additional details provided in the [report](https://github.com/ryanneph/VideoStabilizer/blob/master/submission/neph_ee236c_report.pdf).
