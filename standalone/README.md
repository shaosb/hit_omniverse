# Standalone script of hit_omniverse

## train.py
train a RL-based policy of an environment set up by yaml

## hit_play.py
play trainned weights for an environment  

## hit_sim.py
test an example environment

## TODO list
* finding bugs for environment and training policy
* testing whether position action and effort action can execute well
* redefine PD parmaters and dof stiffness and damping
* training 12 dof robot with position action
* training 12 dof robot with effort action
* training all dof robot with position action (with SimpleMLP and SimpleLSTM)
* training all dof robot with effort action (with normalized NormalizedSimpleMLP ans SimpltLSTM)
* testing action space and observation space
* redefine reward function
