<h1 align="center">HumanDIL: Decouped Imitation for Whole-body <br> Humanoid Natural Locomotion</h1>

# Abstract
Humanoid robot control based on imitation learning has become a research hotspot for enhancing complex terrain locomotion capabilities due to its effectiveness in transferring human motion dexterity and adaptability. We proposes a novel imitation learning-based control framework called Human Decoupled Imitation Learning (HumanDIL). In it, a motion primitive reconstruction-based action redirection mechanism that extracts human "key skill set" and establishes a bipedal motion feature space, significantly improving data utilization and cross-terrain generalization in imitation learning. Then, a hierarchical control architecture where the lower body employs contact-aware end-to-end imitation learning for terrain-adaptive gait generation, while the upper body progressively enhances disturbance robustness through curriculum learning-guided impedance control. This framework strengthens posture coordination under dynamic disturbances through progressive complex terrain training. We evaluate HumanDIL on the HIT-Hu humanoid robot, validating its generalization capabilities and imitation performance across multiple tasks.

# Overview


<div style="display: flex; justify-content: center; align-items: center;">
  <div style="text-align: center; margin: 10px;">
    <img src="gif/wave.gif" width="200"><br>
    <p>Wave</p>
  </div>
  <div style="text-align: center; margin: 10px;">
    <img src="gif/pick.gif" width="200"><br>
    <p>Pick</p>
  </div>
</div>

<p align="center">
    <img src="gif/walk.gif"">
</p>
<center> Walk </center>

<p align="center">
    <img src="gif/run.gif"">
</p>
<center> Run </center>

<p align="center">
    <img src="gif/Jump.gif"">
</p>
<center> Jump forward </center>



# Getting Start

1. Download Isaac Sim from the [website](https://developer.nvidia.com/isaac/sim) with vision >= 4.0.0, then follow the installation instructions.

2. Following the [instructions](https://docs.robotsfan.com/isaaclab/) to install Isaac Lab.

3. Once Isaac Lab is installed, install the external dependencies for this repo:
`pip install -r requirements.txt`

4. Install HIT_omniverse with pip by running:
`pip install -e .`


# License
This project is licensed under the MIT License. Note that the repository relies on third-party code, which is subject to their respective licenses.