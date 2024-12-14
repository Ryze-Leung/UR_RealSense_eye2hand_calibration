
# Robot and Camera Calibration

This repository contains scripts for connecting and calibrating a RealSense D435i camera and a UR robot using RTDE.

---

## Prerequisites

Before running the scripts, make sure to install the required dependencies.

### Dependencies

Install the following Python libraries:

- `numpy`
- `socket`
- `struct`
- `matplotlib`
- `opencv-python`
- `pyrealsense2`
- `ur_rtde`
- `scipy`

You can install these dependencies using `pip`:

```bash
pip install numpy socket struct matplotlib opencv-python pyrealsense2 ur_rtde scipy
```

## Usage

### 1. Test Camera Connection

Run the following script to test the connection with the RealSense D435i camera:

```bash
python realsenseD435i.py
```

### 2. Test Robot Connection

Run the following script to test the connection with the UR robot:

```bash
python connect_robot.py
```

### 3. Perform Calibration

To perform calibration using RTDE, execute the following script:

```bash
python calibrate_rtde.py
```


## 🙌 

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

---

## 📢 

Share this project with your friends and colleagues who might find it useful.

---
# Acknowledgements

Special thanks to Andy Zeng for his contribution to the Value Propagation Gradients (VPG) method. His work has been foundational to this project.
For more details, visit [VPG by Andy Zeng](https://github.com/andyzeng/visual-pushing-grasping).
