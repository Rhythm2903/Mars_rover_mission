# Mars Rover Autonomous Terrain & Obstacle Classifier (Systems Engineering Simulation)

An interdisciplinary autonomous exploration pipeline that integrates **Deep Learning (Computer Vision)**, **Planetary Robotics (Finite State Machines)**, and **Aerospace Economics (Parametric Cost Modeling)** into a unified closed-loop local simulation.

The system utilizes a custom Convolutional Neural Network (CNN) trained on real NASA Martian surface imagery to drive a rover's real-time physical navigation decisions while balancing stringent budget, power, and structural risk constraints.

---

## 🛠️ System Architecture & Cross-Domain Loop

The project breaks away from traditional isolated machine learning models by enforcing a strict systems-engineering feedback loop:

1. **Finance Domain:** User selects a hardware payload configuration (Compute Core, Sensor Suite, Power Source). The system applies a parametric cost model including mass penalties to calculate total Capital Expenditure ($M) and system capability multipliers.
2. **Vision Domain:** Front-facing telemetry cameras capture surface frames. A custom 4-layer deep PyTorch CNN running with CUDA hardware acceleration classifies the terrain into one of 8 authentic Martian geological structures.
3. **Robotics Domain:** The classification feeds a Finite State Machine (FSM) that dictates instantaneous operational protocols (e.g., engaging Rocker-Bogie articulation over Rocks, hard-halting at Craters, or deploying spectrometers on high-value Impact Ejecta).
4. **Optimization Metric:** Performance is evaluated using a core engineering efficiency ratio: **Scientific Return on Investment (ROI)**, defined as Gross Science Points divided by Total Mission Cost.

---

## 📦 Directory Structure

```text
mars_rover_mission/
├── data/raw/                  # 8-Class NASA Martian texture dataset folders
├── src/                       # Core system modular libraries
│   ├── finance.py             # Parametric costing engine & component catalogs
│   ├── vision.py              # 4-Layer deep PyTorch CNN architecture
│   ├── robotics.py            # FSM decision protocols & physics modifiers
│   ├── telemetry.py           # Matplotlib 2D path trajectory tracking engine
│   └── reports.py             # Markdown executive summary generator
├── outputs/
│   ├── models/                # Saved weights (.pth) trained via CUDA
│   ├── telemetry/             # Generated 2D trajectory tracking maps
│   └── financial_reports/     # Itemized mission audit receipts
├── train_vision.py            # Pipeline trainer script
└── deploy_mission.py          # Master closed-loop deployment simulation execution script