function [odn, krn, sat3dPoints, satEdges, camera] = initializePoseEstimationCNN(cameraJSON, chkpt_odn, chkpt_krn)

    camera = jsondecode(fileread(cameraJSON));

    load("tangoPoints.mat", "tango3Dpoints_refined");
    load("tangoEdges.mat", "tangoEdges");
    sat3dPoints = tango3Dpoints_refined;
    satEdges    = tangoEdges;  

    odn = load(fullfile("checkpoints", chkpt_odn), 'net');
    krn = load(fullfile("checkpoints", chkpt_krn), 'net');

end

