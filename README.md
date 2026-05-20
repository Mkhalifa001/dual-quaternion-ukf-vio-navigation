# Dual Quaternion-Based UKF with Visual-Inertial Odometry

This repository contains a MATLAB implementation of a dual quaternion-based Unscented Kalman Filter (UKF) for visual-inertial navigation in GPS-denied environments.

The work was developed as part of my MSc research in Mechanical and Aerospace Engineering at Carleton University. The filter represents the pose using unit dual quaternions and uses visual-inertial measurements for navigation of robotic and aerial systems.

## Overview

The objective of this project is to estimate the motion of a moving platform using inertial and stereo visual measurements. The proposed framework uses a dual quaternion representation to describe the coupled rotational and translational motion of the system.

The implementation includes:

- Dual quaternion-based pose representation
- Unscented Kalman filtering for nonlinear state estimation
- Visual-inertial odometry using stereo image measurements and IMU data
- MATLAB-based simulation and evaluation framework
- Navigation performance evaluation using benchmark datasets

## Method Summary

The filter estimates the navigation state by combining high-rate inertial measurements with visual information obtained from stereo image data. The pose is represented using unit dual quaternions, which provide a compact representation of rigid-body rotation and translation. The uncertainty is handled through a local pose-error representation, while the nominal pose remains on the dual quaternion manifold. This avoids directly adding Euclidean perturbations to the constrained dual quaternion state.

## Code 

The main MATLAB implementation is located in:

```text
src/dqukf_vio_navigation.m
