# 2AMC15-2023-DIC Group 10
In this repository, you will find all the code which we used for the second assignment. We used two different agents,
value iteration and Deep Q-Learning, 


## Quickstart

Before running:
Make sure you have the project running in a virtual environment (recommended venv) and install the requirements.txt file:

```commandline
python3 -m venv venv
source venv/bin/activate        #macOS/Linux
venv\Scripts\activate           #Windows
pip install -r requirements.txt
```

We implemented the following agents and coresponding train files:
1. Value Iteration agent --> `train_value_iteration.py` for both training and evaluation.
2. Deep Q-Learning agent --> `train_DQN.py` for both training and evaluation and `evaluate_DQN.py` for just evaluation. Running this requires a machine with CUDA cores.

In order to run each you simply need to run each of the above mentioned `.py` files in a terminal with the environment activated.
```commandline
$ python train_value_iteration.py
$ python train_DQN.py
$ python evaluate_DQN.py
```
You can also use the `train_optimal_paths.py` file to compare our agent's performance to the optimal path that the agent could take.
```commandline
$ python train_optimal_paths.py
```

### Agent results
The train files will store their results as CSV files in the `experiments` folder. 
Each evaluation of the agent will also result in an image of the path and a text file in the `results` folder.

## File structure and changes to the provided repository
We kept the same overall file structure and added some things. We added a folder for storing the trained DQN models, 
a folder for storing the results of the experiments and a folder with the configurations of our dynamic environment.
We also created a second environment class, `EnvironmentDQN`, which is customised for the DQN agent. This environment
also includes the functionality to have moving objects in the environment. 

## Requirements

- python ~= 3.10
- numpy >= 1.24
- tqdm ~= 4
- pygame ~= 2.3
- flask ~= 2.2
- flask-socketio ~= 5.3
- pillow ~= 9.4
- colorcet ~=3.0
- pandas >= 2.0
- pytorch