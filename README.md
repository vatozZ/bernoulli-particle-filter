# Bernoulli Particle Filter Simulation for a Point Target

This repository contains a Python implementation of a **Bernoulli Particle Filter (BPF)** designed for **single-target tracking** in cluttered environments. The simulation models a scenario in which a target appears and disappears within the sensor's field of view (FoV), demonstrating probabilistic existence and state estimation.

## Scenario Description
- The simulation runs for **20 discrete time steps**.
- The target **appears at step 2** and **leaves the FoV at step 15**.
- Measurements are subject to **false alarms (clutter)** and **missed detections**.
- The algorithm maintains **both state and existence probabilities** of the target.

## Algorithm Highlights
- Bayesian recursive estimation
- Bernoulli Random Finite Set (RFS) formulation
- Importance sampling, resampling
- Handles target birth/death and measurement-origin uncertainty

## File Structure

- `bpf_monte_carlo_simulation.py`: Entry point to run the Bernoulli Particle Filter simulation.
- `bernoulli_pf_parameter_1.py` & `bernoulli_pf_parameter_2.py`: Simulations with different initial parameter sets to evaluate the filter’s robustness to initialization.
- `visualize_MAP_estimations.py`: Visualizes the Maximum A-Posteriori (MAP) state estimations from the simulation logs.
- `visualize_OSPA_and_qk.py`: Plots the OSPA (Optimal Sub-Pattern Assignment) metric and compares it with the target existence probability over time.
- `ospa.py`: Contains the OSPA metric class used for evaluating tracking performance.
