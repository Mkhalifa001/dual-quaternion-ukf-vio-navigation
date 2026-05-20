PROJECT: Dual Quaternion UKF-Based Stereo VIO
Author: Mohamed Khalifa
Supervisor: Dr. Hashim Mohamed
Institution: Carleton University
--------------------------------------------------
1. DESCRIPTION
--------------------------------------------------
This project implements a Dual Quaternion Unscented Kalman Filter (DQUKF)
for full 6-DoF pose, velocity, and bias estimation using stereo camera
and IMU measurements from the EuRoC MAV dataset.

The implementation includes:
- Dual Quaternion-based state propagation
- Minimal error-state covariance (15x15)
- 17x1 full state representation (dq, v, bw, ba)
- Landmark-based measurement update

--------------------------------------------------
2. REQUIREMENTS
--------------------------------------------------
- MATLAB R2023a or newer
- Computer Vision Toolbox
- Aerospace Toolbox (for quaternion utilities)

--------------------------------------------------
3. DATASET SETUP
--------------------------------------------------
Download EuRoC MAV dataset and modify the path inside:

DQUKF_twistor.m

Example:
path = 'C:\...\EuRoC Dataset\V1_03_difficult\mav0';

Make sure the following MATLAB (.mat) files are located in the same directory as DQUKF_twistor.m:

• stereoParamsDataset.mat

• seed_data.mat

The program will not execute correctly if these files are missing or placed in a different folder.

--------------------------------------------------
4. PARAMETER TUNING
--------------------------------------------------
Before running the filter, ensure that the covariances below are set 
according to the selected EuRoC dataset.

--------------------------------------------------
A) V1_01_easy
--------------------------------------------------

1) Covariance Initialization:

   Process Noise:
        Q(10:12,10:12) = (sigma_bw^2)*eye(3);
        Q(13:15,13:15) = (sigma_ba^2)*eye(3);

   IMU Noise:
        Qa = blkdiag((sigma_g^2)*eye(3), (sigma_a^2)*eye(3));

   Initial State Covariance:
        P = blkdiag(zeros(3)+eye(3)*10^2 , zeros(3)+eye(3)*10^2 , 1*eye(3) , eye(6)*10^-1);

2) Visual Measurement Noise:
   R_matrix = (z_std+0.098898)^2*eye(pointsNum*3);

--------------------------------------------------
B) V1_02_medium
--------------------------------------------------

1) Covariance Initialization:

   Process Noise:
        Q(10:12,10:12) = (sigma_bw^2)*eye(3);
        Q(13:15,13:15) = (sigma_ba^2)*eye(3);

   IMU Noise:
        Qa = blkdiag((sigma_g^2)*eye(3), (sigma_a^2)*eye(3));

   Initial State Covariance:
        P = blkdiag(zeros(3)+eye(3)*10^-2 , zeros(3)+eye(3)*10^0 , 1*10^1*eye(3) , eye(6)*10^0);

2) Visual Measurement Noise:
   R_matrix = (z_std+0.099538)^2*eye(pointsNum*3);

--------------------------------------------------
C) V1_03_difficult
--------------------------------------------------

1) Covariance Initialization:

   Process Noise:
        Q(10:12,10:12) = (sigma_bw^2)*eye(3);
        Q(13:15,13:15) = (sigma_ba^2)*eye(3);

   IMU Noise:
        Qa = blkdiag((sigma_g^2)*eye(3), (sigma_a^2)*eye(3));

   Initial State Covariance:
        P = blkdiag(zeros(3)+eye(3)*10^-2 , zeros(3)+eye(3)*10^0 , 1*10^1*eye(3) , eye(6)*10^0);

2) Visual Measurement Noise:
   R_matrix = (z_std+0.3)^2*eye(pointsNum*3);

--------------------------------------------------
D) MEKF and QUKF Tuning
--------------------------------------------------

For Multiplicative EKF and quaternion-based UKF:

• State Initialization:
  - Same state initialization as the DQUKF for all benchmark filters.
  - The initial state is based on the first ground truth entry, with deliberately large perturbations applied to the pose and velocity.
  - The applied perturbations include a significant attitude misalignment, a 3.46 m position error, and a 0.37 m/s velocity error.

• Initial State Covariance (Same as DQUKF):
  V1_01_easy:
  P = blkdiag(zeros(3)+eye(3)*10^2 , zeros(3)+eye(3)*10^2 , 1*eye(3) , eye(6)*10^-1);

  V1_02_medium & V1_03_difficult:
  P = blkdiag(zeros(3)+eye(3)*10^-2 , zeros(3)+eye(3)*10^0 , 1*10^1*eye(3) , eye(6)*10^0);

• Measurement Covariance:
  V1_01_easy      → R = (z_std + 0.1)^2 * eye(pointsNum*3);
  V1_02_medium    → R = (z_std + 0.2)^2 * eye(pointsNum*3);
  V1_03_difficult → R = (z_std + 0.3)^2 * eye(pointsNum*3);

Note:
The number of skipped iterations for RMSE computation should match 
the DQUKF settings for each dataset.

--------------------------------------------------
5. SUMMARY
--------------------------------------------------
1) Open MATLAB.
2) Navigate to the project directory.
3) Open DQUKF_twistor.m.
4) Verify dataset path and tuning parameters.
5) Run the script.

The filter will:
- Initialize the state and covariance
- Perform propagation using IMU data
- Apply stereo landmark updates
- Compute and display RMSE
- Generate trajectory and error plots

Ensure that tuning parameters match the selected dataset before execution.








