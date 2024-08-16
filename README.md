# mUAV-MTT-MAPPO
## Table of Contents

- [Background](#Background)
- [Installation](#Installation)
- [Usage](#Usage)

## Background
We use light_MAPPO in GitHub: https://github.com/tinyzqh/light_mappo and implement our own environment to complete the learning process. 

Since the environment is already defined by the code, training can be done directly by running train.py without additional configuration of other environments.

## Installation

- Follow the instructions of project light_MAPPO to complete the installation process. 

- Run train.py and install other required libraries according to the error message

## Usage

- Run train.py for our proposed algorithm. 
- Original MAPPO can be tested by Commenting out the try_collision function usage in env_runner.py. 
- Off-policy baselines can be seen in off-policy part.
- Modify the algorithm hyperparameters by modifying the default values in config.py or passing in new values on command line calls.
- Modify the values in env_core.py to change the simulation environment settings.

## Other hyperparameters not provided in the paper
Variable | Meaning | Value
--- | :--- |:---
r<sub>min</sub> | target minimum distance | 15km
r<sub>max</sub> | target maximum distance | 25km
θ<sub>min</sub> | target minimum angle| 0
θ<sub>max</sub> | target maximum angle | Pi/2
r<sub>v,min</sub> | target minimum distance | 0.05km/Δt
r<sub>v,max</sub> | target maximum distance | 0.15km/Δt
θ<sub>v,min</sub> | target minimum angle| Pi
θ<sub>v,max</sub> | target maximum angle | 3Pi/2
f<sub>i,j,k,1</sub> | reward factor | 1.0×10<sup>13</sup>
f<sub>i,j,k,2</sub> | reward factor | 1.0×10<sup>13</sup>
f<sub>n(i),j,k</sub> | reward factor | 1.0×10<sup>14</sup>
σ<sub>pred</sub> | Standard deviation of position prediction | 0.01km
d<sub>0</sub> | UAV maximum moving distance | 0.2km
d<sub>1</sub> | UAV collision distance| 0.01km
d<sub>2</sub> | target attack range | 0.4km
α| reward weight | 1.0
penalty | penalty for constraint violation | 5000

## Video of the running example in the paper

See A_running_example.avi.