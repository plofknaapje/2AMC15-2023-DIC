"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
import random
from random import randint
import numpy as np
import math
from agents import BaseAgent


class QLearnAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, theta=0.001, epsilon=0.5, alpha=0.4):
        """
        Set agent parameters.

        Args:
            agent_number: The index of the agent in the environment.
            gamma: loss rate.
            theta: minimal change.
            epsilon: epsilon greedy
            alpha: learning rate
        """
        super().__init__(agent_number)
        self.gamma = gamma
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = None
        self.alphas = None
        self.epsilons = None
        self.dirtGrid = np.zeros(4)
        self.epsilon_decay = epsilon
        self.alpha_decay = alpha
        self.state = [0, 0, 0]
        self.new_state = [0, 0, 0]

    def process_reward(self, action: int, reward: float):
        # update the Q function
        self.Q[self.state[0], self.state[1], self.state[2], action] += \
            self.alpha_decay * (reward + self.gamma * np.max(self.Q[self.new_state[0], self.new_state[1], self.new_state[2], :]) -
                           self.Q[self.state[0], self.state[1], self.state[2], action])
        return 0

    def take_action(self, observation: np.ndarray, info: None | dict):

        # TODO: How to initialize the Q table
        if self.Q is None:
            print('Initializing Q')
            self.Q = np.zeros([observation.shape[0], observation.shape[1], 2 ** 4, 4])
            self.epsilons = np.full((observation.shape[0], observation.shape[1]), self.epsilon)
            self.alphas = np.full((observation.shape[0], observation.shape[1]), self.alpha)
            #self.Q[:, 0, :, :] = -1000000  # first column we don't want to visit
            #self.Q[:, -1, :, :] = -1000000  # last column we don't want to visit
            #self.Q[0, :, :, :] = -1000000  # first row we don't want to visit
            #self.Q[-1, :, :, :] = -1000000  # last row we don't want to visit

            # also set all values for inner walls to low values
            #indices_walls = np.argwhere(observation == 2)
            #for idx in indices_walls:
            #    self.Q[idx[0], idx[1], :, :] = -1000000

            # also set values for dirt already high
            #indices_dirt = np.argwhere(observation == 3)
            #for idx in indices_dirt:
            #    self.Q[idx[0], idx[1], :, :] = 5000

        self.state[0] = info['agent_pos'][0][0]
        self.state[1] = info['agent_pos'][0][1]
        self.state[2] = self.dirt_function(observation, self.state)

        # Set alpha and epsilon according to iteration
        try:
            self.epsilon_decay = self.epsilon * (1-info['iteration'])
            self.alpha_decay = self.alpha * (1-info['iteration'])
            # self.epsilons[self.state[0], self.state[1]] = self.epsilons[self.state[0], self.state[1]] - 0.1
            # self.epsilon_decay = self.epsilons[self.state[0], self.state[1]]
            # if self.epsilon_decay < 0.05:
            #     self.epsilon_decay = 0.05
            # self.alphas[self.state[0], self.state[1]] = self.alphas[self.state[0], self.state[1]] - 0.001
            # self.alpha_decay = self.alphas[self.state[0], self.state[1]]
            # if self.alpha_decay < 0.001:
            #     self.alpha_decay = 0.001
        # If in evaluation no iteration can be found
        except:
            self.epsilon_decay = 0
            self.alpha_decay = 0.001

        # TODO: During random moves: Should we train it with bumping into walls or should we avoid it?
        # take action according to epsilon greedy
        if np.random.uniform(0, 1) < self.epsilon_decay:
            action = randint(0, 3)
            #action = self.get_action(self.state, observation)
        else:
            action = np.argmax(self.Q[self.state[0], self.state[1], self.state[2], :])

        # new state
        self.new_state = self.get_new_state(observation, action, self.state)

        return action

    def get_action(self, state, observation):
        valid_moves = []

        if observation[state[0]][state[1]+1] not in [1, 2]: # down
            valid_moves.append(0)

        if observation[state[0]][state[1]-1] not in [1, 2]: # up
            valid_moves.append(1)

        if observation[state[0]-1][state[1]] not in [1, 2]: # left
            valid_moves.append(2)

        if observation[state[0]+1][state[1]] not in [1, 2]: # right
            valid_moves.append(3)

        action = random.choice(valid_moves)

        return action


    def get_new_state(self, observation, action, state):
        action_map = {0: [state[0], state[1] + 1, state[2]],  # down
                      1: [state[0], state[1] - 1, state[2]],  # up
                      2: [state[0] - 1, state[1], state[2]],  # left
                      3: [state[0], state[1] + 1, state[2]],  # right
                      }
        new_state = action_map.get(action, state)

        state[2] = self.dirt_function(observation, state)

        if observation[new_state[0], new_state[1]] in [1, 2]:
            return state
        else:
            return new_state


    # TODO: make flexible for different segmentations of the grid (give number of segmentations as input)
    def dirt_function(self, observation: np.ndarray, state):
        #check which quarter
        height = observation.shape[0]
        width = observation.shape[1]
        if state[0] < height/2:
            if state[1] < width/2:
                quarter = 0
            else:
                quarter = 1
        else:
            if state[1] < width/2:
                quarter = 2
            else:
                quarter = 3

        #check if already dirt free
        if self.dirtGrid[quarter] == 1:
            return self.dirt_byte_converter(self.dirtGrid)
        
        dirty = False



        if quarter == 1:
            for i in range(0, math.floor(height/2)):
                for j in range(0, math.floor(width/2)):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break

        
        if quarter == 2:
            for i in range(0, math.floor(height/2)):
                for j in range(math.ceil(width/2), width):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break

        if quarter == 3:
            for i in range(math.ceil(height/2), height):
                for j in range(0, math.floor(width/2)):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break
                
        if quarter == 4:
            for i in range(math.ceil(height/2), height):
                for j in range(math.ceil(width/2), width):
                    if observation[i][j] == 3:
                        dirty = True
                        break
                if dirty:
                    break
        
        if dirty == False:
            self.dirtGrid[quarter] = 1
        return self.dirt_byte_converter(self.dirtGrid)

        #check if there is dirt
        #update dirt 
        
    def dirt_byte_converter(self, dirt_grid):
        number = 0
        if dirt_grid[0] == 1:
            number += 1
        if dirt_grid[1] == 1:
            number += 2
        if dirt_grid[2] == 1:
            number += 4
        if dirt_grid[3] == 1:
            number += 8
        return number



