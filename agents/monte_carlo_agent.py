import numpy as np
from collections import deque
from agents import BaseAgent


class MCAgent(BaseAgent):
    def __init__(self, agent_number, obs: np.ndarray, gamma=0.5, epsilon=0.4):
        
        super().__init__(agent_number)

        # parameters
        self.max_len_episode = 30
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps_without_cleaning = 20
        self.agent_start_pos=None
        # Keeps track of the starting position, and last found dirt's location.
        # If we have moved too much without finding any more, we'll reset to
        # that location. 
        # Initial value will be set in take_action because we don't know it yet.
        
        # self.reset_location = None

        self.x_size, self.y_size = obs.shape
        self.A = [0, 1, 2, 3]

        # Create a 3D array named Returns, where each entry is a dictionary
        # the dictionary contains the following keys:
        # "score" = the total score of the episode
        # "n" = the number of times the state was visited
        self.Returns = [
            [
                
                [{"score": 0, "n": 0} for i in range(len(self.A))]

                for i in range(self.y_size)
            ]
            for i in range(self.x_size)
        ]

        self.Q = [
            [
                {
                    0: np.random.rand(),
                    1: np.random.rand(),
                    2: np.random.rand(),
                    3: np.random.rand(),
                    # 4: np.random.rand(),
                }
                for i in range(self.y_size)
            ]
            for i in range(self.x_size)
        ]

        self.policy = np.zeros((self.x_size, self.y_size))
        # create initial policy
        self.update_policy()

        # to keep track of (s,a,r) of each step in episode
        self.episode = deque()
        self.episode_rewards = deque()

    def process_reward(self, obs: np.ndarray, reward: float):
        """
        Check if terminated 
        (we assume that happens if charging station reached) 
        Or maximum number of steps reached in episode.
        If so -> update Q and policy and reset episode.
        """
        # Add reward obtained to the list with rewards
        self.episode_rewards.append(reward)

        # # If we find dirt somewhere, this will be the starting position
        # # of the next world (= (x,y) grid with current d value)
        # if reward >= 5:
        #     x, y = info["agent_pos"][self.agent_number]
        #     self.reset_location = (int(x), int(y))

        # Check if terminated (=end of episode or charging station reached with all dirt cleaned)
        if reward == 10 or len(self.episode) >= self.max_len_episode:
            # update Q and policy
            self.update_Q()
            self.update_policy()

            # If we've not found anything in the entire episode, go back to last place where
            # dirt was found
            # need_to_reset_location = all(reward < 5 for reward in self.episode_rewards)

            # if need_to_reset_location:

            # reset episode
            self.reset_episode()

    def reset_episode(self):
        """
        Reset the deques that keep track of the episode state and rewards.
        """
        self.episode = deque()
        self.episode_rewards = deque()

    def update_Q(self):
        """
        After an episode is finished, or we've reached the charging station, update the Q values for all
        state action pairs seen in the episode.
        """
        # Create a new 3D array named newQ to keep track of the state action pairs seen in the episode
        # initially all entries are False. As soon as we see a state action pair, we set the corresponding
        # cell to a dict which keeps track of the discounted reward
        newQ = [
            [
                [False for i in range(len(self.A))]
                for i in range(self.y_size)
            ]
            for i in range(self.x_size)
        ]

        # create deque that keeps track of the state action pairs seen in the episode
        state_action_pairs_seen = deque()

        for index, ((currentX, currentY), currentA) in enumerate(
            self.episode
        ):
            currentX, currentY, currentA = (
                int(currentX),
                int(currentY),

                int(currentA),
            )
            # This loop is designed in such a way that we have to loop over
            # the state action pairs seen in the episode only once. As we loop
            # over them, we update the discounted rewards for each state action pair.
            # we do this by keeping a dict for each seen state action pair, which contains
            # the current discounted reward value and the current gamma exponent value.
            currentReward = self.episode_rewards[index]

            # Update all discounted rewards for previously spotted state action pairs
            for prevX, prevY, prevA in state_action_pairs_seen:
                discounted_reward_dict = newQ[prevX][prevY][prevA]

                new_gamma_exponent = discounted_reward_dict["gamma_exponent"] + 1
                discounted_reward_dict["gamma_exponent"] = new_gamma_exponent

                discounted_reward_dict["discounted_reward"] += (
                    self.gamma**new_gamma_exponent * currentReward
                )

            # Create dict for this state action pair if is the first time we are seeing it
            # in this episode
            if not newQ[currentX][currentY][
                currentA
            ]:  # means that this is the first time we are seeing this state action pair in this episode
                newQ[currentX][currentY][currentA] = {
                    "discounted_reward": currentReward * self.gamma,
                    "gamma_exponent": 1,
                }
                state_action_pairs_seen.append((currentX, currentY, currentA))

        # update Returns: score, n
        # and update Q[x][y][a] with the new AVG(!) discounted reward
        for x, y, a in state_action_pairs_seen:
            self.Returns[x][y][a]["score"] += newQ[x][y][a]["discounted_reward"]
            self.Returns[x][y][a]["n"] += 1

            self.Q[x][y][a] = (
                self.Returns[x][y][a]["score"] / self.Returns[x][y][a]["n"]
            )

    def update_policy(self, optimal=False):
        """
        Update the policy for each state value.

        This is done by looking at the Q values for each state and choosing between
        the action with the highest Q value with probability 1-epsilon + epsilon/|A|
        and a random different action with probability epsilon/|A|.

        Args:
            optimal (bool, optional): If True, we take the optimal policy. If false,
                we take the epsilon greedy policy. Defaults to False.
        """
        # loop over all x,y in self.Q
        for x in range(self.x_size):
            for y in range(self.y_size):
                Q_values = self.Q[x][y]
                # find the key that corresponds to the max value
                max_key = max(Q_values, key=Q_values.get)

                # If training is over, we want to take the optimal policy
                if optimal:
                    action = max_key
                else:
                    action_probs = len(self.A) * [self.epsilon / len(self.A)]

                    action_probs[max_key] = (
                        1 - self.epsilon + self.epsilon / len(self.A)
                    )

                    action = np.random.choice(a=4, p=action_probs)

                # set the policy at x,y,d to the chosen action
                self.policy[x, y] = action

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Return the action that should be taken by the agent, based on the current
        policy and state (= (x,y)).
        """
        # Get the x,y values of the state
        x, y = info["agent_pos"][self.agent_number]
        x, y = int(x), int(y)

        # If this is the initial location, set this as place to reset to
        # self.reset_location = (x, y)


        # For current state, get the action based on the current policy
        next_action = self.policy[x][y]

        # Log the state action pair in self.episode
        self.episode.append(((x, y), next_action))

        # return the action we take
        return next_action
