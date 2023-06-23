# 2AMC15-2023-DIC Group 10
In this repository, you will find all the code which we used for the second assignment. We used two different agents,
Value Iteration and Deep Q-Learning. Both were run on our 5 dirt warehouse and we tested our DQN agent more by also 
having it run in a dynamic environment. 

## Repository setup

Before running:
Make sure you have the project running in a virtual environment (recommended venv) and install the requirements.txt file.
To do this, open a terminal window in this folder and run the following commands. 

### Conda
```commandline
conda create -n DIC10           #conda users
conda activate DIC10            #conda users
pip install -r requirements.txt
```

### Venv
```commandline
python3 -m venv venv
source venv/bin/activate        #macOS/Linux
venv\Scripts\activate           #Windows
pip install -r requirements.txt
```

### Verification
Run the following command to check if all the required packages were installed:
```commandline
pip list
```

## 
We implemented the following agents and coresponding train files:
1. Value Iteration agent --> `train_value_iteration.py` for both training and evaluation.
2. Deep Q-Learning agent --> `train_DQN.py` for both training and evaluation and `evaluate_DQN.py` for just evaluation. 
The Deep Q-Learning agent uses PyTorch and will use CUDA cores if your machine has them.

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
The train files will store their results as CSV files in the `experiments` folder. We already preloaded those. Be aware
that since we use a random factor in our environment (sigma=0.3), the results of the experiments are not deterministic.
Each evaluation of the agent will also result in an image of the path and a text file in the `results` folder.
The DQN models will be saved in the `DQN_models` folder after training. We already pretrained those.

## File structure and changes to the provided repository
We kept the same overall file structure and added some things.
We added two features to the given environmnet: moving objects and agent vision.
The moving objects can be programmed using a json file, examples can be found in `dynamic_env_config`, specifically `test.json`.

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