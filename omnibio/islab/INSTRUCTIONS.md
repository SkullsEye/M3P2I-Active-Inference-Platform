# Isaac Lab Simulation Setup

## Overview
This project simulates a robot navigating an environment using Active Inference Planning (AIP) within NVIDIA Isaac Lab (Isaac Sim).

**What we did:**
1.  **Created Phase 3:** Migrated the simulation logic from Phase 2 (Matplotlib-based) to Phase 3 (Isaac Sim-based).
2.  **Integrated Isaac Sim:** Built a script (`run_isaac.py`) that initializes the Isaac Sim world, spawns a robot (blue cube), a goal (red sphere), and obstacles (red cylinders).
3.  **Connected Planner:** Linked the existing `ActiveInferencePlanner` and `CostPlanner` to drive the robot in the 3D physics environment.
4.  **Enhanced Scene:** Added a ground plane and distant lighting for better visibility.
5.  **Configured Environment:** Resolved display issues by targeting `DISPLAY=:1` for local execution.

## How to Run

1.  **Activate Conda Environment:**
    Ensure you are in the correct environment (e.g., `unitree_sim_env`) where Isaac Lab is installed.
    ```bash
    conda activate unitree_sim_env
    ```

2.  **Set Display (if needed):**
    If the window doesn't appear, check your display variable:
    ```bash
    echo $DISPLAY
    ```
    If it returns `:1`, run:
    ```bash
    export DISPLAY=:1
    ```

3.  **Launch Simulation:**
    Run the Python script using the python interpreter from your environment:
    ```bash
    python ~/m3p2i-aip/omnibio/phase3/run_isaac.py
    ```

## File Structure
-   `phase3/run_isaac.py`: Main entry point for the Isaac Sim simulation.
-   `common/`: Shared code for planners and behavior trees.
