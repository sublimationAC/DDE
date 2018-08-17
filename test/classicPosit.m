function [rotation, translation] = classicPosit(imagePoints, objectPoints, focalLength, center)
%
% Usage:  [rotation, translation] = classicPosit(imagePoints, objectPoints, focalLength, center)
% Return rotation and translation of world object 
% given world points and image points
% Inputs:
% imagePoints is a matrix of size nbPts x 2
% objectPoints is a matrix of size nbPts x 3
% focalLength is the focal length of the camera in PIXELS
% center is a row vector with the two components of the image center
%
% If center is not given, assume the image coordinates 
% are w.r.t. image reference at image center
%
% Outputs:
% rot: a 3 x 3 rotation matrix of scene with respect to camera
% trans: 3 x 1 translation vector from projection center of camera to FIRST POINT in list of object points
%
% This version is a translation of the Mathematica code published in IJCV 15 (1995)
% Example of input: 
%
% cube = [ 0 0 0; 10 0 0; 10 10 0; 0 10 0; 0 0 10; 10 0 10; 10 10 10; 0 10 10]
% cubeImage = [0 0; 80 -93; 245 -77; 185 32; 32 135; 99 35; 247 62; 195 179]
% focalLength = 760
% [rot, trans] = classicPosit(cubeImage, cube, focalLength)
%
% Outputs: 
% rot = [ 0.4901  0.8506  0.1906; -0.5695  0.146  0.8088; 0.6600 -0.5049  0.5563];
% trans = [0.0; 0.0; 40.0392]; 
% when computation stops after 5 iterations

% Copyright (c) 1993-2003 Daniel DeMenthon and University of Maryland
% All rights reserved

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

msg = nargchk(3, 4, nargin);
if (~isempty(msg))
	error(strcat('ERROR:', msg));
end

clear msg;

if nargin == 3 % image coordinates are already centered, center is not given
	center = [0, 0];
end

%=======================================================================================

imagePoints(:,1) = (imagePoints(:,1) - center(1));
imagePoints(:,2) = (imagePoints(:,2) - center(2));

converged = 0;
count = 0;
imageDifference = 99;

nbPoints = size(imagePoints, 1);

objectVectors = objectPoints - ones(nbPoints,1) * objectPoints(1,:);
objectVectors
objectMatrix = pinv(objectVectors) % pseudoinverse

oldSOPImagePoints = imagePoints;

while ~converged
	if count == 0
		imageVectors = imagePoints - ones(nbPoints,1) * imagePoints(1,:);
	else % count>0, we compute a SOP image first for POSIT
		correction = (1 + (objectVectors * row3')/translation(3));
		spreadCorrection = correction * ones(1, 2);
		SOPImagePoints = imagePoints .* spreadCorrection;
		diffImagePoints = abs(round(SOPImagePoints - oldSOPImagePoints))
		imageDifference = ones(1, nbPoints) * diffImagePoints * ones(2,1)  % add up all coordinates
		oldSOPImagePoints = SOPImagePoints;
		imageVectors = SOPImagePoints - ones(nbPoints,1) * SOPImagePoints(1,:)
	end % else

	IJ = (objectMatrix * imageVectors)'
	IVect = IJ(1,:); JVect = IJ(2,:);
	ISquare = IVect*IVect';
	JSquare = JVect*JVect';
	scale1 = sqrt(ISquare); scale2 = sqrt(JSquare);
	row1 = IVect/scale1; row2 = JVect/scale2;
	row3 = cross(row1, row2);
	rotation = [row1; row2; row3]
	scale = (scale1 + scale2)/2.0;
	translation = [imagePoints(1,:), focalLength]/scale
		
	converged = (count > 0 & imageDifference < 1)
	count = count + 1
	disp(' ');
	disp(' ======================================================= ');
	disp(' ');

end % while

%%test data
%     41.7988     39.5977
%     33.0155     48.1345
%      19.635     48.5756
%     8.90857     44.9377
%     19.6343     43.9858
%     31.0244      43.046
%    -42.5142      40.539
%    -35.1484     50.6449
%    -18.7285     51.0164
%    -7.99127     42.5838
%    -20.9935     44.8427
%    -33.0524     44.7015
%    -42.5717     25.4084
%    -28.5628     33.7748
%     -14.943     27.1595
%    -28.1737     25.0191
%     40.8977     27.3543
%     27.4731     33.5804
%     14.2916     27.8617
%     28.4458     25.4084
%    -5.68854     34.9066
%    -9.10229     19.4991
%    -15.4844     2.46584
%    -11.7781    -4.64537
%   -0.148712   0.0917664
%      15.795    -4.51912
%     15.7239     1.70552
%     8.75818     19.6777
%     7.38858     34.0763
%    -7.25079     1.85912
%     7.86343     1.94215
%    -21.3634    -33.5457
%     -18.973    -21.3225
%    -11.6354    -15.0618
%   -0.155762    -14.0889
%     10.7403    -15.0618
%     16.5758     -18.153
%     22.6084    -29.8487
%     19.7047    -41.2991
%     13.2695    -48.9162
%     2.76266    -51.6402
%    -6.57623    -50.0841
%    -15.1823    -45.2644
%    -8.98007    -44.8564
%     2.56808    -47.3601
%      11.763    -43.8465
%     11.0244    -21.0239
%   0.0388184    -18.9532
%    -12.6082    -21.4823
% -0.00341797     -32.602
%   -0.263885     8.70103
%    -35.9561     31.6345
%    -35.1778     25.4084
%    -21.9471     25.6033
%    -20.9743     32.9967
%     19.3009     32.8018
%     20.2737     26.5758
%     34.4772     25.9921
%     34.0881     31.4401
% 
%   -0.155319   -0.196573   -0.117928
%   -0.122275   -0.202457   -0.138402
%  -0.0703791   -0.212317    -0.15357
%   -0.045445   -0.192763    -0.15218
%  -0.0831619   -0.191762   -0.144808
%    -0.11491   -0.189787   -0.138528
%    0.166264   -0.190818   -0.099298
%    0.123435   -0.212578   -0.131626
%   0.0709847    -0.20756   -0.148431
%   0.0383834   -0.187887   -0.146913
%     0.11319   -0.193974   -0.136106
%   0.0713183   -0.191273   -0.140996
%    0.140024   -0.150562   -0.106945
%   0.0891935    -0.17121   -0.132544
%   0.0469165   -0.148664   -0.131428
%   0.0955024   -0.138307   -0.130764
%   -0.149064   -0.155156   -0.103498
%   -0.102238   -0.172368   -0.135763
%  -0.0628161   -0.149919   -0.130662
%   -0.101318   -0.141879   -0.132338
%  0.00663469   -0.159861   -0.160375
%   0.0402687  -0.0971145   -0.162176
%   0.0497906  -0.0574806   -0.165663
%   0.0437425   -0.025354   -0.188062
%  -0.0076449  -0.0224399   -0.226046
%  -0.0630918   -0.026498   -0.180662
%   -0.061012   -0.056219   -0.176621
%  -0.0436971   -0.104141   -0.170619
%  -0.0264088   -0.162712    -0.15368
%   0.0159821  -0.0340607   -0.218447
%  -0.0427064  -0.0350848   -0.208974
%   0.0637852   0.0626376   -0.165935
%   0.0426968   0.0446144   -0.188555
%   0.0200896   0.0367372   -0.200563
%  -0.0112115   0.0344711   -0.206729
%   -0.036098   0.0344515   -0.200788
%   -0.064508   0.0428893   -0.184755
%  -0.0888063   0.0576384   -0.157458
%  -0.0619057   0.0701735   -0.176815
%  -0.0470364    0.076336   -0.189316
%  -0.0212099   0.0859561   -0.199864
%   0.0126039   0.0816349   -0.199208
%     0.04247   0.0735101   -0.180601
%   0.0140097   0.0613537   -0.193607
%  -0.0199347   0.0654714   -0.198159
%  -0.0379179   0.0634419   -0.191646
%  -0.0444614   0.0523667   -0.191165
% -0.00944232   0.0532676   -0.202057
%   0.0200286   0.0563057   -0.192332
%  -0.0202703    0.056138    -0.19143
% -0.00813358  -0.0492357   -0.243715
%    0.117087   -0.166559   -0.125599
%    0.117052   -0.144651   -0.121583
%   0.0739499   -0.143117   -0.131686
%   0.0658134   -0.166934   -0.131188
%  -0.0823002    -0.16702   -0.132579
%   -0.082921   -0.146135   -0.132772
%   -0.129907   -0.146206    -0.11838
%   -0.123318   -0.168948   -0.128681
