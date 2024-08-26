# HIT_omniverse

A simulation environment for HIT_HU robot based on ISAAC LAB.

Including a dynamic random simulation environment generator implemented in ISAAC SIM, a motion controller based on imitation learning, and adequate external sensing interfaces for future development.

## Getting Start

Download Isaac Sim from the [website](https://developer.nvidia.com/isaac/sim) with vision >= 4.0.0, then follow the installation instructions.

Following the [instructions](https://docs.robotsfan.com/isaaclab/) to install Isaac Lab.

Once Isaac Lab is installed, install the external dependencies for this repo:
`pip install -r requirements.txt`

Install HIT_omniverse with pip by running:
`pip install -e .`

## Formation
> hit_omniverse
>> algo  
>> configs   // configs for robots and simulation environments
>> extension   // class of environment and robots
>>> mdp   // reward, observation, command function and dataset setting
>> dataset   // dataset
>> standalone   // script files
>> utils   // helper function

## Data Preparation

## License
This project is licensed under the MIT License. Note that the repository relies on third-party code, which is subject to their respective licenses.