# Heat Transfer Simulation in Brain Tissue Using FEniCSx

## Overview

This code sets up and runs a computational simulation of heat diffusion within a small square domain representing a section of brain tissue. It solves the transient heat equation using the finite element method (FEM) implemented with FEniCSx.

The simulation models how heat generated internally (e.g., by a localized source) spreads through brain tissue over time, accounting for the tissue’s physical properties such as thermal conductivity and volumetric heat capacity.

---

## What This Code Does

- **Domain and Mesh:**  
  Defines a 1 cm × 1 cm square domain discretized into a triangular mesh with 50×50 resolution.

- **Physical Parameters:**  
  Sets brain tissue thermal conductivity (`k = 0.5 W/m·K`) and volumetric heat capacity (`rho_c = 3.8e6 J/m³·K`), from which the thermal diffusivity is calculated.

- **Initial and Boundary Conditions:**  
  The temperature is initially uniform at 37°C (normal brain temperature). Dirichlet boundary conditions fix the temperature at the domain edges to 37°C throughout the simulation, simulating a constant-temperature environment.

- **Heat Source:**  
  Introduces a Gaussian-shaped heat source near the center of the domain, representing localized heat generation such as might be caused by a neural recording electrode during operation.

- **Time Stepping:**  
  Uses an implicit Euler scheme with small time steps (`dt = 0.01 s`) to evolve the temperature over 1000 steps, simulating transient heat diffusion.

- **Solution and Output:**  
  At each time step, the code assembles and solves the linear system for the temperature field, printing the maximum temperature and writing the results to an XDMF file for visualization.

---

## Context and Larger Project

This simulation forms the initial setup for a larger computational project aimed at understanding and visualizing how neural recording electrodes heat up brain tissue during operation. 

**Why this matters:**  
Neural electrodes generate heat that can affect tissue health and recording quality. Accurately modeling heat diffusion around these devices helps neural engineers design safer, more effective electrodes and predict thermal effects in neural interfaces.

By combining fundamental physics of heat transfer with finite element analysis, this project offers hands-on experience with computational modeling and neurotechnology. Future work will extend this basic setup by incorporating realistic electrode geometries, dynamic boundary conditions, and coupled biophysical effects.

---

## How to Use

1. Install dependencies:  
   - FEniCSx (version 0.9 or compatible)  
   - `mpi4py` and `petsc4py`  
   - `numpy`

2. Run the script using an MPI-enabled Python environment (e.g., `mpiexec -n 4 python heat_sim.py`).

3. Visualize the results using Paraview or another XDMF-compatible visualization tool by loading `results.xdmf`.

---
