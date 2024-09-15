clear all; close all; clc

% STEP 1: Initialize everything - this should be done outside of the demo
% loop because it only needs to be done once.
cameraJSON = "cameraNEW.json"; % Path to camera intrinsics json file for plotting wireframe
chkpt_odn  = "odn_speedplusv2_net_checkpoint__50000__2024_09_14__01_13_49.mat"; % Name of ODN checkpoint, shouldn't need to change this
chkpt_krn  = "krn_speedplusv2_net_checkpoint__49950__2024_09_15__07_58_04.mat"; % Name of KRN checkpoint, shouldn't need to change this

[odn, krn, sat3dPoints, satEdges, camera] = initializePoseEstimationCNN(cameraJSON, chkpt_odn, chkpt_krn);

% STEP 2: Pose estimation -This should be put in the loop after an image is
% acquired

%%% For testing only, loading in an image, in the demo, this should just be
%%% the acquired TRON image
I = imread("TRON_images_NASA/img000137.jpg");

estimatePose(I, odn, krn, sat3dPoints, satEdges, camera)