# HRNET Pose Sample App

## Table of Contents
1. [About](#about)
2. [Project Status](#project-status)
3. [Setup](#setup)
4. [Run](#run)
5. [Directory Structure](#directory-structure)

## About
This app is provided as a pose detection sample using the open source HRNET-POSE model from Qualcomm AI Hub. The application uses ONNX runtime (ORT) to enable the model to run cross-platform.

On the Snapdragon X Elite, the model is optimized to leverage the Neural Processing Unit (NPU) at inference runtime. Elsewhere, it will run using the CPU.

## Project Status
This sample python app has only been validated using Windows 11 Enterprise Snapdragon(R) X Elite

## Setup
Follow these steps to setup the app for your platform.

### Snapdragon X Elite
   1. git clone repo
   2. Create virtual environment
      ```
      >> python3.11 -m virtual_env env_sample_app_hrnet
      ```
   3. Activate virtual environment
      ```
      >> env_sample_app_hrnet/Scripts/activate.ps1 (Windows: Validated)
      >> src env_sample_app_hrnet/bin/activate (Mac: Validated)
      >> src env_sample_app_hrnet/bin/activate (Linux: Not Validated)  # Will not work via WSL due to camera binding issue within WSL
      ```
   4. Install dependencies
      ```
      >> pip install -r requirements.txt
      ```
   5. Download model from AI Hub 
      https://aihub.qualcomm.com/compute/models/hrnet_pose?domain=Computer+Vision&useCase=Pose+Estimation

   6. Transfer model to qnn_sample_apps/models/
      ```
      >> mv Downloads/hrnet_pose.onnx qnn_sample_apps/models/
      ```

### Mac/Linux
Coming Soon

## Run
<!-- **To run:** </br> -->
```
>> python ./src/hrnet_pose/main.py (from root directory)
>> python ./src/hrnet_pose/main.py --system windows --model hrnet_pose --processor cpu --camera 1 --available_cameras False
```

## Contributing
We welcome contributions to this repository! Please refer to our [contributing guide](CONTRIBUTING.md) for how to contribute.

## Directory Structure
```
qnn_sample_apps
├─ .gitignore
├─ dll.json
├─ executioner.json
├─ models
│  └─ README.md
├─ models.json
├─ notebooks
│  └─ sample_app_hrnet.ipynb
├─ pyproject.toml
├─ README.md
├─ requirements.in
├─ requirements.txt
├─ scripts
│  └─ directory_information.txt
├─ setup.py
├─ src
│  ├─ hrnet_pose
│  │  ├─ main.py
│  │  ├─ model_inference.py
│  │  ├─ model_loader.py
│  │  └─ README.md
│  └─ __init__.py
└─ tests
   └─ test_module.py

```
