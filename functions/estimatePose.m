function estimatePose(I, odn, krn, sat3dPoints, satEdges, camera)

import clib.opencv.*;
import vision.opencv.util.*;

% This function will take in the image, the odn and krn models and the
% points/edges for tango and will estimate tango's pose and then produce a
% plot with the predicted wireframe overlayed on the image
if length(size(I)) < 3
        I = repmat(I, [1, 1, 3]);
end
imgSize = size(I, [1, 2]);
imgSize = imgSize([2, 1]);
odnInputSize = [416, 416];
krnInputSize = [224, 224];

% ============================== 1. ODN ============================== %
% Process image
img = imresize(I, odnInputSize);
img = im2single(img);

% Detect bbox with IoU threshold 0.5
bbox_pr = detect(odn.net, img, 'Threshold', 0.5); % [xmin, ymin, w, h] 
bbox_pr = double(bbox_pr);                        % (pix., resized)

% Prediction to [xmin, ymin, w, h] (pix. org.)
bbox_pr = bbox_pr ./ [odnInputSize, odnInputSize] .* [imgSize, imgSize];

% ============================== 2. KRN ============================== %
% Process image with predicted bbox
% Recall: bbox [xmin, ymin, w, h] (pix. org.)
x = bbox_pr(1) + bbox_pr(3) / 2;
y = bbox_pr(2) + bbox_pr(4) / 2;
w = bbox_pr(3);
h = bbox_pr(4);

[xmin, ymin, xmax, ymax] = getSquareRoI(x, y, w, h, imgSize([2, 1]), false);
roi = [xmin, ymin, xmax - xmin, ymax - ymin]; % [xmin, ymin, w, h] (pix. org.)
img = imcrop(I, roi);
img = imresize(img, krnInputSize);

% Predict keypoints
img  = im2single(img);
kpts = predict(krn.net, img); % [1 x 22] normalized keypoints (x1, y1, x2, y2, ...)

% Predicted keypoints to [2 x 11] normalized keypoints
kpts = reshape(double(kpts), [2, 11]);

% Keypoints to pixels in original image
kpts_pr(1,:) = kpts(1,:) * (xmax - xmin) + xmin;
kpts_pr(2,:) = kpts(2,:) * (ymax - ymin) + ymin;

% ============================== 3. PnP ============================== %

% EPnP using MATLAB-OpenCV Interface
% - Rp: world -> VBS
% - Tp: VBS -> world in VBS
kpts2d = kpts_pr';
kpts3d = sat3dPoints';

% Create Input and Output clib arrays and set options for cv.solvePnP:
[kpts2dMat,kpts2d_clibArr]        = createMat(kpts2d);
[kpts3dMat, kpts3d_clibArr]       = createMat(kpts3d);
[cameraMat,cameraMat_clibArr]     = createMat(camera.cameraMatrix);
[distCoeffMat, distCoeff_clibArr] = createMat("Input"); % empty array (zero distortion)
[rvecMat, rvec]                   = createMat; % Output rotation vector
[tvecMat, tvec]                   = createMat; % Output translation vector
useExtrinsicGuess                 = false;
flags                             = 1; % corresponds to EPnP as solver

% use OpenCV solvePnP function
RetVal = cv.solvePnP(kpts3d_clibArr,...
                     kpts2d_clibArr,...
                     cameraMat_clibArr,...
                     distCoeff_clibArr,...
                     rvec,...
                     tvec,...
                     useExtrinsicGuess,...
                     flags);

% Convert rvec and tvec back to MATLAB arrays and convert rvec to
% quaternion 
[R_prMat, R_pr_clibArr] = createMat;
cv.Rodrigues(rvec, R_pr_clibArr); %Convert rvec to rotation matrix
R_pr = getImage(R_pr_clibArr); % Convert rotation matrix to MATLAB array
q_vbs2target_pr = dcm2quat(R_pr'); % Rotation matrix to quaternion
r_Vo2To_vbs_pr  = getImage(tvec)';

% ============================== Plot ============================== %
imshow(I); hold on;
%     scatter(kpts_pr(1,:), kpts_pr(2,:), 16, 'gx');
%     xmin = bbox_pr(1); ymin = bbox_pr(2); xmax = xmin + bbox_pr(3); ymax = ymin + bbox_pr(4);
%     xmin = bbox_gt(1); ymin = bbox_gt(2); xmax = xmin + bbox_gt(3); ymax = ymin + bbox_gt(4);
plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], 'Color', 'y', 'LineWidth', 1.5);
plotWireframe(camera, satEdges, q_vbs2target_pr, r_Vo2To_vbs_pr, 'Color', 'g', 'LineWidth', 1.5);
waitforbuttonpress

end