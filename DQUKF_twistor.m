% Note: All the built-in function like quatmultiply, quatinv, etc. will not work unless given row vectors

clc; clear; close all;

tic

path        = 'C:\Users\Mohamed Khalifa\Desktop\Research\Navigation Filters\EuRoC Dataset\V1_02_medium\mav0';
[IMURead, Groundtruth, cam0_Read, cam1_Read] = dataRead(path);

% Load stereo camera intrinsic parameters (calibrated using MATLAB)
load('stereoParamsdataset.mat');
seed_data = load('seed_data.mat');
rng(seed_data.seed_val)

% transfromation from body frame to cam0 frame 
TC02B = [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975;
          0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768;
          -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949;
          0.0, 0.0, 0.0, 1.0];

%% UKF parameters
alpha = 1e-4;         % spread of sigma points
kappa = 0;            % secondary scaling parameter
beta = 2;             % incorporates prior knowledge (Beta = 2 is optimal for Gaussian)
Nx = 17;              % state dimension
Nx_eff = Nx - 2;      % Considering only the 6 independent coordinates in the dual quaternions (3 rotation + 3 translation)
dt = 0.005;           % time step
Nxa = 17 + 6;         % augmented state dimension
xa = zeros(6,1);      % mean of process and measurement noises with [nw' ; na']
Nxa_eff = Nxa - 2;    % Considering only the 6 independent coordinates in the dual quaternion

Lambda = 3-Nxa_eff;
gamma = sqrt(Nxa_eff + Lambda);

% Weights
Wm = [Lambda / (Nxa_eff + Lambda), repmat(1 / (2 * (Nxa_eff + Lambda)), 1, 2 * Nxa_eff)];
Wc = Wm;
Wc(1) = Wm(1) + (1 - alpha^2 + beta);

%% Keyframe initialization
% camIdxHist = [];
% camFrameId = [];
% featLeft   = {};
% featRight  = {};

numFeatures = 60;
idx = 1;
[imageLeft,imageRight] = stereoRead(cam0_Read,cam1_Read,idx,stereoParams); % Reads and rectifies stereo image pair
pointTracker = vision.PointTracker('BlockSize',[9 9],'MaxBidirectionalError',1);
[pointsKeyR,pointsKeyL,pointTracker] = funPointTracker(imageLeft,imageRight,numFeatures,pointTracker);
points3D_body  = funPoints3D(pointsKeyL,pointsKeyR,stereoParams,TC02B); % 3x60

%% Simulation Parameters
TimerIMU  = IMURead(:,1);  % seconds 
TimerCam0 = cam0_Read.time; % seconds 

% True variables initialization
g  = 9.81;
e3 = [0,0,1]';
bw = mean(Groundtruth(:,12:14)); % averaged gyro bias from groundtruth
ba = mean(Groundtruth(end-100:end,15:17)); % averaged accel bias from groundtruth
p  = Groundtruth(1,2:4)'; % initial position
R  = quat2rotm(Groundtruth(1,5:8)); % Rotation matrix from initial quaternion
v  = Groundtruth(1,9:11)'; % initial velocity

Tfinal     = ceil(length(TimerIMU)*0.9); % 16701*0.9 = 15031 --> ceil(15031) = 15031

% ====== Initializing state ==========
% axis = [1; 1; -1];
% axis = axis / norm(axis);
% angle = deg2rad(30);                   % attitude error
% q_error = axang2quat([axis' angle]);   % 1x4
% q_pert  = quatmultiply(q_error, rotm2quat(R));
% q_pert  = q_pert / norm(q_pert);
qhat = rotm2quat(R)'; 
qhat = 2*qhat+8*[2 , 3 , -2 , 1]';
qhat = qhat/norm(qhat); 
p_pert = p + [2, 2, -2]';     % meters
v_pert = v + [0.3, -0.2, 0.1]';   % m/s

xhatstore = zeros(Nx, Tfinal);
q_dual = rigid_body_transform(qhat', p_pert'); % ensure qhat = 1x4 and p = 1x3
q_dual = normalize_dual_quaternion(q_dual); % normalize dual quaternion --> 1x8
xhat = [q_dual' ; v_pert; bw'; ba']; % [8x1 dual quaternion; 3x1 velocity; 3x1 gyro bias; 3x1 accel bias] = 17x1 vector
xhatstore(: , 1) = xhat; % storing initial xhat in the first column

% Initial Covariance Matrices
% ==== Use this for all V1 datasets to ensure convergence ====
sigma_g  = 2.4e-3;     
sigma_a  = 2.8e-2;     
sigma_bw = 2.7e-4;     % gyro bias random walk
sigma_ba = 4.2e-2;     % accel bias random walk

Q = zeros(15);
Q(10:12,10:12) = (sigma_bw^2)*eye(3);
Q(13:15,13:15) = (sigma_ba^2)*eye(3);
Qa = blkdiag((sigma_g^2)*eye(3), (sigma_a^2)*eye(3));

% P = blkdiag(zeros(3)+eye(3)*10^2 , zeros(3)+eye(3)*10^2 , 1*eye(3) , eye(6)*10^-1); % Use this for V1_01_easy
P = blkdiag(zeros(3)+eye(3)*10^-2 , zeros(3)+eye(3)*10^0 , 1*10^1*eye(3) , eye(6)*10^0); % Use this for V1_02_medium & V1_03_difficult
x = xhat; % initializing state vector (17x1)

% Sigma points Storage
Xi = zeros(2*Nxa_eff + 1, Nxa_eff);   % sigma points [43, 21]
Sigma_pts_f = zeros(2*Nxa_eff+1, Nx); % transformed sigma points through f(x) [43, 17]

% landmarks
points3D_world = R * points3D_body + p; % inertial frame landmarks

Time = (TimerIMU(1:Tfinal)-TimerIMU(1)); % [1x15031] total time (approx 75s) with increments of 0.005s

%% Loop
Restimate  = zeros(3,3,Tfinal);
Pestimate  = zeros(3,Tfinal);
Vestimate  = zeros(3, Tfinal);
drotv      = zeros(3, Tfinal);
normP      = zeros(1, Tfinal);
normV      = zeros(1, Tfinal);
normbw     = zeros(1, Tfinal);
Numlandmark    = zeros(size(cam0_Read.files));

for k=1:1:Tfinal % for k = 1,2,3... 15031
    if mod(floor(100*k/Tfinal), 5) == 0 % mod(floor(0.006653*k), 5) @ k = 15031 --> mod(100,5) = 0
        clc
        percent_complete = floor(100 * k / Tfinal);
        elapsed_time = toc; % Measure the elapsed time
        
        % Estimate the total time based on current progress
        estimated_total_time = elapsed_time / (k / (Tfinal - 1));
        estimated_time_left = estimated_total_time - elapsed_time;
        
        % Convert elapsed_time to hours, minutes, and seconds
        elapsed_hours = floor(elapsed_time / 3600);
        elapsed_minutes = floor(mod(elapsed_time, 3600) / 60);
        elapsed_seconds = round(mod(elapsed_time, 60)); % Round to nearest second
        
        % Convert estimated_time_left to hours, minutes, and seconds
        estimated_hours = floor(estimated_time_left / 3600);
        estimated_minutes = floor(mod(estimated_time_left, 3600) / 60);
        estimated_seconds = round(mod(estimated_time_left, 60)); % Round to nearest second
        
        % Display the progress and estimated time left
        disp(['Complete  ' num2str(percent_complete) '%']);
        disp(['Elapsed time: ' sprintf('%02d:%02d:%02d', elapsed_hours, elapsed_minutes, elapsed_seconds)]);
        disp(['Estimated time left: ' sprintf('%02d:%02d:%02d', estimated_hours, estimated_minutes, estimated_seconds)]);
    end
    
    q_truth       = Groundtruth(k,5:8);
    R_truth       = quat2rotm(q_truth);
    p_truth       = Groundtruth(k,2:4)';
    dq_truth      = rigid_body_transform(q_truth, p_truth');
    v_truth       = Groundtruth(k,9:11)';
    Restimate(:,:,k) = quat2rotm(xhat(1:4)');
    Pestimate(:,k) = recover_translation(xhat(1:8)')';
    Vestimate(:,k) = xhat(9:11);
    xhatstore(:,k) = xhat;

    normP(1,k) = norm(p_truth - Pestimate(:,k));
    normV(1,k) = norm(v_truth - Vestimate(:,k));
    normbw(1,k) = norm(bw'-xhat(12:14));

    % Display every 20 iterations
    dq_gt = rigid_body_transform(Groundtruth(k,5:8), Groundtruth(k,2:4));
    dq_pose_err = dual_quat_multiply(dq_inverse(dq_gt), xhat(1:8)');
    dq_pose_err = normalize_dual_quaternion(dq_pose_err);
    twistor_err = dq_to_twistor(dq_pose_err);   % 6x1 twistor error

    q_err  = quatnormalize(quatmultiply(Groundtruth(k,5:8), quatinv(xhat(1:4)')))'; % 4x1
    if q_err(1) < 0
        q_err = -q_err;
    end
    axang = quat2axang(q_err');
    drotv(:,k) = axang(1:3)' * axang(4);   % 3x1 rotation-vector error

    error_p          = Groundtruth(k,2:4)' - recover_translation(xhat(1:8)')'; % 3x1
    error_v          = Groundtruth(k,9:11)' - xhat(9:11); % 3x1
    error_bw         = Groundtruth(k,12:14)' - xhat(12:14); % 3x1
    error_ba         = Groundtruth(k,15:17)' - xhat(15:17); % 3x1

    if mod(k, 20) == 0
        fprintf('Iteration: %d\n', k);
        fprintf('Orientation error: [%.4f, %.4f, %.4f]\n', drotv(:,k));
        fprintf('Position Error: [%.4f, %.4f, %.4f]\n', error_p);
        fprintf('Velocity Error: [%.4f, %.4f, %.4f]\n', error_v);
        fprintf('Gyro Bias Error: [%.4f, %.4f, %.4f]\n', error_bw);
        fprintf('Accel Bias Error: [%.4f, %.4f, %.4f]\n', error_ba);
        fprintf('----------------------------\n');
    end

    %          SIGMA POINT GENERATION 

    dT    = TimerIMU(k+1)-TimerIMU(k); % dT = 0.005
    omega = IMURead(k,2:4)' ; % from IMU
    a     = IMURead(k,5:7)' ; % from IMU

    % Nominal state components
    dq_nom = xhat(1:8);      % 8x1 nominal unit dual quaternion
    v_nom = xhat(9:11);     % 3x1 nominal velocity
    bg_nom = xhat(12:14);   % 3x1 nominal gyro bias
    ba_nom = xhat(15:17);   % 3x1 nominal accel bias

    x_err = zeros(Nx_eff,1);      % Nx_eff = 15
    x_aug = [x_err; xa]; % R^21
    P_aug = blkdiag(P, Qa); % (15x15) + (6x6) = 21x21
    gamma_Pa = (Nxa_eff + Lambda) * P_aug; % simply 3*P_aug (scaled)

    % Using SVD or lower cholesky decomposition chol(gamma_Pa, 'lower')
    [U, S, V] = svd(gamma_Pa);
    sPa = U * sqrt(S) * V'; % [21 + (3-21)]*P = (21x21)

    % Sigma points in augmented local error coordinates
    Xi(1, :) = x_aug;

    for i = 1:Nxa_eff
        Xi(i+1 , :) = x_aug + sPa(:,i);
        Xi(Nxa_eff + i+1 , :) =  x_aug - sPa(:,i);
    end
    Sigma_pts = zeros(2*Nxa_eff + 1, Nxa);   % 43 x 23
    
    for i = 1:(2*Nxa_eff + 1)
        delta_twistor = Xi(i, 1:6);
        delta_v       = Xi(i, 7:9);
        delta_bg      = Xi(i, 10:12);
        delta_ba      = Xi(i, 13:15);
        noise_bg      = Xi(i, 16:18);
        noise_ba      = Xi(i, 19:21);

        dq_err = twistor_to_dq(delta_twistor'); % 1x8

        % Compose with nominal pose
        dq_sigma = dual_quat_multiply(dq_nom', dq_err);
        dq_sigma = normalize_dual_quaternion(dq_sigma)';
        v_sigma  = v_nom  + delta_v';
        bg_sigma = bg_nom + delta_bg';
        ba_sigma = ba_nom + delta_ba';

        Sigma_pts(i,:) = [dq_sigma; v_sigma; bg_sigma; ba_sigma; noise_bg'; noise_ba']'; % 43x23

    end


    %          PREDICTION STEP
    % Process each sigma point through the process model (each row represent a sigma point for all states)
    for i = 1: 2*Nxa_eff + 1 % for i = 1, 2, ... 43
        Sigma_pts_f(i, :) = dynamics(Sigma_pts(i, :)', dT , omega , a , g , e3)'; % sgm_pts(i,:)' = 17x1, sgm_pts_f --> 43x17
    end
    % Convert propagated dual quaternion points to twistor
    dq_ref = Sigma_pts_f(1,1:8);   % predicted reference dual quaternion
    delta_twistor_pred = zeros(2*Nxa_eff+1, 6);
    for i = 1:(2*Nxa_eff + 1)
        dq_i = Sigma_pts_f(i,1:8);
        delta_dq = dual_quat_multiply(dq_inverse(dq_ref), dq_i);
        delta_dq = normalize_dual_quaternion(delta_dq);
        delta_twistor_pred(i,:) = dq_to_twistor(delta_dq)';   % 43x6
    end

    % Applying weights to find predicted state and covariance
    x_pred = zeros(Nx, 1); % 17x1 a priori state estimate
    P_pred = zeros(Nx_eff, Nx_eff); % 15x15

    twistor_pred = Wm*delta_twistor_pred; % (1x43)(43x6) = 1x6
    dq_corr = twistor_to_dq(twistor_pred');
    dq_pred = dual_quat_multiply(dq_ref, dq_corr);
    dq_pred = normalize_dual_quaternion(dq_pred);
    x_pred(1:8) = dq_pred';
    x_pred(9:end) = Wm*Sigma_pts_f(:, 9:end); % Non-quaternion elemntings weighting x_pred(5:end) = 13x1 --> x_pred = 17x1
    x = x_pred; % Updating x for next iteration to conserve temporal relations

    P_pred = P_pred + Q; % Add process noise covariance (15x15)+(15x15) = (15x15)

    % for loop to find x_dev = sigma_pts_f(i,:) - x_pred: subtraction of dquaternion requires special handling
    x_dev = zeros(Nx_eff, 1); % 15x1
    for  n = 1:2*Nxa_eff + 1 % for n = 1,2, ... 43
        x_dev(1:6)   = delta_twistor_pred(n,:)' - twistor_pred';
        x_dev(7:end) = Sigma_pts_f(n, 9:end)' - x_pred(9:end); % For standard or non-quaternion elements (9x1) - (9x1) = 9x1
        P_pred = P_pred + Wc(n) * (x_dev * x_dev'); % Wc(n) just take individual elements in the row/column vector
    end
    P_pred = symmetrizeCovariance(P_pred);
    P = P_pred;
  
     % ====== Visual update ======
    if TimerIMU(k+1) >= TimerCam0(idx+1,1)
        idx = idx +1; % the visual update occurs the idx+1 time instant
        [imageLeft,imageRight]             = stereoRead(cam0_Read,cam1_Read,idx,stereoParams);
        [pointsNewL,pointsNewR,isFoundand] = stereoPointTrack(imageLeft,imageRight,pointTracker);
        % tracked features locations
        pointsNewR = pointsNewR(isFoundand,:);
        pointsNewL = pointsNewL(isFoundand,:);

        % camIdxHist(end+1) = idx;
        % camFrameId(end+1) = k;
        % featLeft{end+1}   = pointsNewL;
        % featRight{end+1}  = pointsNewR;

        pointsI   = points3D_world(:,isFoundand);
        pointsB   = funPoints3D(pointsNewL,pointsNewR,stereoParams,TC02B);
        [pointsI,pointsB,pointsNum] = pointsSelection(pointsI,pointsB);
        Numlandmark(idx)= size(pointsI,2);

        if (pointsNum>=3)
            z_std = 0; 
            % R_matrix = (z_std+0.098898)^2*eye(pointsNum*3);  % Use this for V1_01_easy
            R_matrix = (z_std+0.099538)^2*eye(pointsNum*3);  % % Use this for V1_02_medium
            % R_matrix = (z_std+0.3)^2*eye(pointsNum*3); % Use this for V1_03_difficult

            %          MEASUREMENT PREDICTION
            Sigma_pts_h = zeros(2*Nxa_eff+1, 3*pointsNum); % transformed sigma points through h(x) --> [43, 3*pointNum] = [43, 24]
            for i = 1: 2*Nxa_eff + 1
                q = Sigma_pts_f(i, 1:4); % 1x4    
                p = recover_translation(Sigma_pts_f(i, 1:8))'; % 3x1
                R = quat2rotm(q); % 3x3 - body to world
                z_pred = reshape(R'*(pointsI - p) , [] , 1); % (3x3)*(3x10)--> reshape(3x10) = 24x1 
                Sigma_pts_h(i, :) = z_pred'; % each row represent the propagated sigma point through the measurement model
            end
        
            % Compute the predicted measurement and its covariance
            Pz = zeros(3*pointsNum, 3*pointsNum); % 24x24
            zk = (Wm*Sigma_pts_h)'; % (1x43)(43x24) = (1x24)' = 24x1
            Pz = Pz + R_matrix; % 24x24 + (24x24 measurement noise)
            for j = 1: 2*Nxa_eff + 1
                z_dev = Sigma_pts_h(j, :)' - zk; % 24x1 - 24x1 = 24x1
                Pz = Pz + Wc(j) * (z_dev * z_dev'); % 24x24
            end
            Pz = symmetrizeCovariance(Pz);

            % Compute cross-covariance
            Pxz = zeros(Nx_eff, 3*pointsNum); % 15x24
            x_r = zeros(Nx_eff,1); % 15x1
            for m = 1:2*Nxa_eff + 1
                x_r(1:6) = delta_twistor_pred(m,:)' - twistor_pred';
                x_r(7:end) = Sigma_pts_f(m, 9:end)' - x(9:end); % For standard or non-quaternion elements (9x1) - (9x1) = 9x1
                z_r = Sigma_pts_h(m, :)' - zk; % 24x1 - 24x1 = 24x1
                Pxz = Pxz + Wc(m) * (x_r * z_r'); % The matrix is normally not symmetric, thus, don't symmetrize it!
            end

            %          UPDATE STEP

            % Kalman gain
            K = Pxz / Pz; % 15x24

            % Measurement vector
            z = reshape(pointsB + z_std*randn(size(pointsB)), [], 1); % pointsB = 3x8 --> z = 24x1
            Innov = K * (z - zk); % (15x24)(24x1) = 15x1 
        
            % State Update
            delta_twistor_upd = Innov(1:6);
            dq_perturbation = twistor_to_dq(delta_twistor_upd);
            dq_sum = dual_quat_multiply(x(1:8)', dq_perturbation);
            dq_sum = normalize_dual_quaternion(dq_sum);
            plus = x(9:end) + Innov(7:end); % x = 17x1, Innov = 15x1 --> 9x1
            x = [dq_sum'; plus]; % 17x1
        
            % Covariance Update
            P = P - K * Pz * K'; % 15x15 - (15x24)(24x24)(24x15) = 15x15
            P = symmetrizeCovariance(P);
        end
        % Creat new Keyframe if pointsNum is less than 6 
        minFeatures = 6;
        if pointsNum <= minFeatures %size(find(isFoundand==1), 1) < minFeatures       
            [pointsKeyR,pointsKeyL,pointTracker] = funPointTracker(imageLeft,imageRight,numFeatures,pointTracker);                 
            points3D_body  = funPoints3D(pointsKeyL,pointsKeyR,stereoParams,TC02B);
            % points3D_world = quat2rotm(x(1:4)') * points3D_body + recover_translation(x(1:8)')';
            points3D_world = R_truth * points3D_body + p_truth;  %inertial frame landmarks 
        end
    end
       
    xhat = x;

end

release(pointTracker);
% save("DQUKF_animation_data.mat", ...
%     "cam0_Read","cam1_Read", ...
%     "camIdxHist","camFrameId", ...
%     "featLeft","featRight", ...
%     "Pestimate","Restimate", ...
%     "Groundtruth", ...
%     "drotv","normP","normV","Time");
% Plotting
PGrdtruth  = Groundtruth(1:Tfinal,2:4)'; % 3 x 15031
VGrdtruth  = Groundtruth(1:Tfinal,9:11)'; % 3 x 15031
errorR     = drotv; % 3 x 15031
errorP     = Pestimate - PGrdtruth; % 3 x 15031
errorV     = Vestimate - VGrdtruth; % 3 x 15031
errorRnorm = vecnorm(drotv); % 1 x 15031
errorPnorm = normP; % 1 x 15031
errorVnorm = normV; % 1 x 15031

% Define navigation error as: e =  errorRnorm + errorPnorm + errorVnorm \in R --> Find RMSE of e
e = errorRnorm(:, 4000:end) + errorPnorm(:, 4000:end) + errorVnorm(:, 4000:end); % 1 x 15031 --> skip 2000 for initializing using GT or skip 4000 as with perturbed initialization
RMSE       = sqrt(mean(e.^2)); 
RMSE_R     = sqrt(mean(errorRnorm.^2)); 
RMSE_P     = sqrt(mean(errorPnorm.^2)); 
RMSE_V     = sqrt(mean(errorVnorm.^2)); 

% Trajectory
figure
set(gcf,'Color','w')
set(gcf,'Units','inches','Position',[1 1 10 5.5])   % larger figure for document export
set(gcf,'Renderer','Painters')

axBig = axes('Units','normalized','Position',[0.06 0.10 0.60 0.82]);

plot3(axBig, Pestimate(1,1000:end), Pestimate(2,1000:end), Pestimate(3,1000:end),'k-','LineWidth',2.5)
hold(axBig,'on')
plot3(PGrdtruth(1,1000:end), PGrdtruth(2,1000:end),PGrdtruth(3,1000:end),'g-','linewidth',2.5)

xlabel(axBig,'x (m)','FontSize',20)
ylabel(axBig,'y (m)','FontSize',20)
zlabel(axBig,'z (m)','FontSize',20)
grid(axBig,'on')

% -------- Draw body axes along estimated trajectory --------
step  = 2000;    % draw every N samples
scale = 0.2;     % axis length in meters

for k = 1000:step:Tfinal

    p = Pestimate(:,k);
    R = Restimate(:,:,k);

    % x-axis (red)
    quiver3(axBig, p(1), p(2), p(3), ...
        scale*R(1,1), scale*R(2,1), scale*R(3,1), ...
        'r--','LineWidth',2);

    % y-axis (green)
    quiver3(axBig, p(1), p(2), p(3), ...
        scale*R(1,2), scale*R(2,2), scale*R(3,2), ...
        'g--','LineWidth',2);

    % z-axis (blue)
    quiver3(axBig, p(1), p(2), p(3), ...
        scale*R(1,3), scale*R(2,3), scale*R(3,3), ...
        'b--','LineWidth',2);
end

set(axBig,'FontSize',18,'LineWidth',1.5)

% MAGNITUDE OF ERROR
pR = uipanel('Units','normalized','Position',[0.70 0.10 0.28 0.82], ...
    'BorderType','none','BackgroundColor','w');

tR = tiledlayout(pR,3,1,'TileSpacing','compact','Padding','compact');

% First subplot
ax1 = nexttile(tR,1);
plot(ax1, Time, vecnorm(drotv), 'r-','LineWidth',2)
ylabel(ax1,'$|\tilde{R}|_I$','Interpreter','latex','FontSize',18)
xlabel(ax1,'$t(s)$','Interpreter','latex','FontSize',18)
xlim(ax1,[0 Time(end)])

% Second subplot
ax2 = nexttile(tR,2);
plot(ax2, Time, errorPnorm, 'r-','LineWidth',2)
ylabel(ax2,'$\|p-\hat{p}\|$','Interpreter','latex','FontSize',18)
xlabel(ax2,'$t(s)$','Interpreter','latex','FontSize',18)
xlim(ax2,[0 Time(end)])

% Third subplot
ax3 = nexttile(tR,3);
plot(ax3, Time, normV(1,:), 'r-','LineWidth',2)
ylabel(ax3,'$\|v-\hat{v}\|$','Interpreter','latex','FontSize',18)
xlabel(ax3,'$t(s)$','Interpreter','latex','FontSize',18)
xlim(ax3,[0 Time(end)])

set([ax1 ax2 ax3], ...
    'FontSize',18, ...
    'LineWidth',1.5, ...
    'TickLabelInterpreter','latex')



% ERROR PLOTS
figure;
subplot(3,3,1); % X component rotation
plot(Time , drotv(1,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta q_{k,1}$','interpreter','latex','FontSize',16);
grid on;
subplot(3,3,2); % X component position
plot(Time , errorP(1,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta p_{k,x}$','interpreter','latex','FontSize',16);
grid on;
subplot(3,3,3); % X component velocity
plot(Time , errorV(1,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta v_{k,x}$','interpreter','latex','FontSize',16);
grid on;

subplot(3,3,4); % y component rotation
plot(Time , drotv(2,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta q_{k,2}$','interpreter','latex','FontSize',16);
grid on;
subplot(3,3,5); % y component position
plot(Time , errorP(2,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta p_{k,y}$','interpreter','latex','FontSize',16);
grid on;
subplot(3,3,6); % y component velocity
plot(Time , errorV(2,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta v_{k,y}$','interpreter','latex','FontSize',16);
grid on;

subplot(3,3,7); % z component rotation
plot(Time , drotv(3,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta q_{k,3}$','interpreter','latex','FontSize',16);
grid on;
subplot(3,3,8); % z component position
plot(Time , errorP(3,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta p_{k,z}$','interpreter','latex','FontSize',16);
grid on;
subplot(3,3,9); % z component velocity
plot(Time , errorV(3,:), 'b-','linewidth',2)
set(gca,'FontSize',12)
xlabel('Time [s]','FontSize',16);
ylabel('$\delta v_{k,z}$','interpreter','latex','FontSize',16);
grid on;
% set(gcf,'Units','inches')
% set(gcf,'Position',[1 1 13.5 7])   % width=12in, height=10in
% set(gcf,'PaperPositionMode','auto')
% exportgraphics(gcf,'figure.pdf','ContentType','vector')

% POSITION TRACKING
figure('Position',[22   562   560   420]);
subplot(3,1,1); % X component
plot(Time , PGrdtruth(1,:), 'r-','linewidth',1), hold on;
plot(Time , Pestimate(1,:), 'b--','linewidth',1);
title('X Component');
xlabel('Time');
ylabel('X');
legend('Ground Truth', 'Estimate');
grid on;
hold off;

subplot(3,1,2); % Y component
plot(Time , PGrdtruth(2,:), 'r-','linewidth',1), hold on;
plot(Time , Pestimate(2,:), 'b--','linewidth',1);
title('Y Component');
xlabel('Time');
ylabel('Y');
legend('Ground Truth', 'Estimate');
grid on;
hold off;

subplot(3,1,3); % Z component
plot(Time , PGrdtruth(3,:), 'r-','linewidth',1), hold on;
plot(Time , Pestimate(3,:), 'b--','linewidth',1);
title('Z Component');
xlabel('Time');
ylabel('Z');
legend('Ground Truth', 'Estimate');
grid on;
hold off;

%% Functions
function [imu, groundtruth, cam0, cam1] = dataRead(path)

    % === IMU ===
    fid = fopen([path '\imu0\data.csv']);
    imu_raw = textscan(fid, '%f,%f,%f,%f,%f,%f,%f', 'HeaderLines', 1);
    fclose(fid);
    imu = cell2mat(imu_raw);
    imu(:,1) = imu(:,1) * 1e-9;   % ns → s
    
    % === Groundtruth ===
    fileID      = fopen([path '\state_groundtruth_estimate0\data.csv']);
    grdtruth_raw = textscan(fileID, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', 'HeaderLines', 1);
    fclose(fileID);
    groundtruth = cell2mat(grdtruth_raw);
    groundtruth(:,1) = groundtruth(:,1) * 1e-9;   % ns → s

    % === cam0 ===
    fid = fopen([path '\cam0\data.csv']);
    cam0_raw = textscan(fid, '%f,%s', 'HeaderLines', 1);
    fclose(fid);

    cam0.time = cam0_raw{1} * 1e-9;
    cam0.files = string(cam0_raw{2});
    cam0.img_dir = [path '\cam0\data'];

    % === cam1 ===
    fid = fopen([path '\cam1\data.csv']);
    cam1_raw = textscan(fid, '%f,%s', 'HeaderLines', 1);
    fclose(fid);

    cam1.time = cam1_raw{1} * 1e-9;
    cam1.files = string(cam1_raw{2});
    cam1.img_dir = [path '\cam1\data'];

    % Aligning readings
    intTime     = intersect(imu(:,1),groundtruth(:,1),'sorted'); % finding intersecting timestamps betweem IMU readings and Ground truth
    timestart   = find(intTime(1,1)==imu(:,1));
    timeend     = find(intTime(end,1)==imu(:,1));
    imu         = imu(timestart:timeend,:);
    timestart   = find(intTime(1,1)==groundtruth(:,1));
    timeend     = find(intTime(end,1)==groundtruth(:,1));
    groundtruth = groundtruth(timestart:timeend,:);

    timestart   = find(cam0.time(:,1)<groundtruth(1,1), 1, 'last' );  % last camera timestamp before groundtruth starts
    timeend     = find(cam0.time(:,1)>groundtruth(end,1),1);          % first camera timestamp after groundtruth ends
    cam0.time   = cam0.time(timestart:timeend,:);
    cam0.files  = cam0.files(timestart:timeend,:);
    cam1.time   = cam1.time(timestart:timeend,:);
    cam1.files  = cam1.files(timestart:timeend,:);
end

function normalized = normalize_dual_quaternion(x)
    % Extract real and dual parts
    p = x(1:4);
    q = x(5:8);

    % Normalize real part
    p_norm = norm(p);
    p_normalized = p / p_norm;
    
    % Compute the dot product and apply normalization formula
    dot_product = dot(p_normalized, q);
    q_normalized = (q - p_normalized * dot_product) / p_norm;
    
    normalized = [p_normalized, q_normalized];
end

function dq = rigid_body_transform(q_r, t)
    real_part = q_r;
    q_t = [0, t]; % assuming t is in world frame
    dual_part = quatmultiply(q_t, q_r) / 2;
    dq = [real_part, dual_part];  
end

function product = dual_quat_multiply(x1, x2)
    % Extract real and dual parts
    p1 = x1(1:4);  q1 = x1(5:8);
    p2 = x2(1:4);  q2 = x2(5:8);
    
    % Dual quaternion multiplication
    real_part = quatmultiply(p1, p2);
    dual_part = quatmultiply(p1, q2) + quatmultiply(q1, p2);
    
    % Combine and ensure consistent representation
    product = [real_part, dual_part];
end

function S = Skew(v)
    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
        -v(2), v(1), 0];
end

function next_state = dynamics(cur_state , dT , omega , a , g , e3)
    dq = cur_state(1:8); % 8x1 dual quaternion
    real_part = dq(1:4); % rotation of the dual quaternion
    Rhat = quat2rotm(real_part');
    vhat = cur_state(9:11);    
    bw = cur_state(12:14); % omega bias
    ba = cur_state(15:17); % acceleration bias
    bw_w = cur_state(18:20); % omega noise
    ba_w = cur_state(21:23); % acceleration noise
 
    % Solve the discrete linear dynamics for the velocity
    v_dyn = -g*e3 + Rhat*(a-ba-ba_w);
    v_next = vhat + dT * v_dyn;

    % dual quaternion dynamics
    omega_ = omega - bw - bw_w; % correct omega readings without bias and noise effects
    v_body = Rhat' * vhat; % convert vhat from inertial to body frame

    % finding Gammas
    Gamma_w = [0 , -omega_';omega_ , -Skew(omega_)];
    Gamma_v = [0 , -v_body';v_body , -Skew(v_body)];

    % Construct 8X8 matrix
    M = [Gamma_w, zeros(4); Gamma_v, Gamma_w];

    % discrete dual quaternion dynamics using matrix exponential
    dq_next = expm(dT*1/2*M)*dq; % 8x1
    dq_hat = normalize_dual_quaternion(dq_next'); % 1x8
    if dq_hat(1) < 0
            dq_hat = -dq_hat;
    end
    
    bw_next = bw + dT * bw_w;   
    ba_next = ba + dT * ba_w;  

    next_state = [dq_hat'; v_next; bw_next; ba_next];
end

function dq = twistor_to_dq(tw)

    tw = tw(:);
    
    p = tw(1:3);
    b = tw(4:6);
    
    p2 = p.' * p;
    den = 1 + p2;
    
    q0 = (1 - p2) / den;
    qv = (2 * p) / den;
    
    qr = [q0; qv];
    
    px = [   0   -p(3)  p(2);
           p(3)   0    -p(1);
          -p(2)  p(1)   0   ];
    
    Pi = 0.25 * ( (1 - p.'*p) * eye(3) - 2*px + 2*(p*p.') );
    
    rO = Pi \ b;
    
    rq0 = 0;
    rqv = rO;
    
    a0 = rq0;   av = rqv;
    b0 = qr(1); bv = qr(2:4);
    
    d0 = a0*b0 - dot(av,bv);
    dv = a0*bv + b0*av + cross(av,bv);
    
    qd = 0.5 * [d0; dv];
    
    dq = [qr, qd];

end

function tw = dq_to_twistor(dq)

    dq = dq(:);
    
    qr = dq(1:4);
    qd = dq(5:8);
    
    if qr(1) < 0
        qr = -qr;
        qd = -qd;
    end
    
    qr_conj = [qr(1); -qr(2:4)];
    
    a0 = qd(1);   av = qd(2:4);
    b0 = qr_conj(1); bv = qr_conj(2:4);
    
    t0 = a0*b0 - dot(av,bv);
    tv = a0*bv + b0*av + cross(av,bv);
    
    t_quat = 2 * [t0; tv];
    rO = t_quat(2:4);
    
    q0 = qr(1);
    qv = qr(2:4);
    
    denom = 1 + q0;
    
    if abs(denom) < 1e-12
        p = -qv / (1 - q0);
    else
        p = qv / denom;
    end
    
    px = [   0   -p(3)  p(2);
           p(3)   0    -p(1);
          -p(2)  p(1)   0   ];
    
    Pi = 0.25 * ( (1 - p.'*p) * eye(3) - 2*px + 2*(p*p.') );
    
    b = Pi * rO;
    
    tw = [p; b];

end

function dq_inv = dq_inverse(dq)
    % input normalized unit quaternion
    qr = dq(1:4);
    qd = dq(5:8);

    qr_conj = quatconj(qr);
    qd_inv  = -quatmultiply(quatmultiply(qr_conj, qd), qr_conj);

    dq_inv = [qr_conj, qd_inv];
end

function t = recover_translation(dq)
    real_part = dq(1:4);
    dual_part = dq(5:8);
    translation = 2 * quatmultiply(dual_part, quatinv(real_part)); % for small translations
    t = translation(2:4);  % Extract x, y, z translation
end

function P_sym = symmetrizeCovariance(P)
    % Symmetrize the covariance matrix P
    % Ensure P is a square matrix
    if size(P, 1) ~= size(P, 2)
        error('Input matrix P must be square.');
    end
    
    % Symmetrize P by averaging it with its transpose
    P_sym = 0.5 * (P + P');
end

function [imgL,imgR] = stereoRead(cam0, cam1, idx, stereoParams)
    % read and rectify with stereoParams
    imgL = imread(fullfile(cam0.img_dir, cam0.files(idx)));
    imgR = imread(fullfile(cam1.img_dir, cam1.files(idx)));
    [imgL,imgR] = rectifyStereoImages(imgL,imgR,stereoParams);
end

function [pointsKeyR,pointsKeyL,pointTracker] = funPointTracker(imageLeft,imageRight,numFeatures,pointTracker)
    points          = detectMinEigenFeatures(imageRight); % Shi–Tomasi corner detector
    %points          = detectORBFeatures(imageRight); % ORB feature detector
    pointsKeyR      = points.Location;
    
    release(pointTracker);

    % Initialize the tracker with the initial point locations and the initial video frame.
    initialize(pointTracker, pointsKeyR, imageRight);

    % track features on the left image of keyframe
    [pointsKeyL, isFound] = pointTracker(imageLeft);

    % outlier removal
    [pointsKeyL, isFound] = outlierRemoval(pointsKeyL,isFound);
    
    pointsNewR = pointsKeyR(isFound,:);
    pointsNewL = pointsKeyL(isFound,:);
     
    % choose high socres munFeatures for efficiency
    if size(pointsNewR, 1)>numFeatures
        randIndex  = randperm(size(pointsNewR, 1),numFeatures);
        pointsKeyR = pointsNewR(randIndex,:); 
        pointsKeyL = pointsNewL(randIndex,:); 
    end
    
    % reload tracker
    setPoints(pointTracker,pointsKeyR); 
end

function points3D = funPoints3D(pointsFeaturesL,pointsFeaturesR,stereoParams,TC02B)
    % Finding the 3D position of features in the body frame
    positionCam0 = 0.001*triangulate(pointsFeaturesL,pointsFeaturesR,stereoParams);
    positionCam0 = positionCam0';
    R = TC02B(1:3,1:3);
    p = TC02B(1:3,4);
    points3D = R*positionCam0 + p;
end

function [pointsNewL,pointsNewR,isFoundand] = stereoPointTrack(imageLeft,imageRight,pointTracker) 
    % features tracking on new stereo images
    [pointsNewR, isFoundR]  = pointTracker(imageRight); 
    [pointsNewL, isFoundL]  = pointTracker(imageLeft);
    
    % outlier removal
    [pointsNewR, isFoundR] = outlierRemoval(pointsNewR,isFoundR);
    [pointsNewL, isFoundL] = outlierRemoval(pointsNewL,isFoundL);
    
    % find features in both images
    isFoundand = and(isFoundR,isFoundL);        
end

function [points,isFound]=outlierRemoval(points,isFound)
    points2 = points(isFound);
    pointsmean = mean(points2,1);
    temp = points2-pointsmean;
    dist = sqrt(diag(temp*temp'));
    distmean = mean(dist);
    diststd  = std(dist);
    index = abs(dist-distmean)<=diststd;
    points2= points2(index,:);
    isFound = ismember(points(:,1),points2(:,1));
end

function [pointsI,pointsB,pointsNum] = pointsSelection(pointsI,pointsB)
    pointsIC   = mean(pointsI,2); % center of landmarks
    pointsBC   = mean(pointsB,2); % center of landmarks
    tempI = sqrt(diag((pointsI - pointsIC)'*(pointsI - pointsIC)));
    tempB = sqrt(diag((pointsB - pointsBC)'*(pointsB - pointsBC)));
    index = abs(tempI-tempB)<=0.01;
    
    pointsI = pointsI(:,index);
    pointsB = pointsB(:,index);
    
    pointsNum = size(pointsI,2);
end
