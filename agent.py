import numpy as np
import utils
import math

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        # generating a state s_{t+1} based off of the new
        s_prime = self.generate_state(environment)

        # TODO: write your function here

        if self.a != None and self.s != None:
            # obtains a reward r
            if dead:
                r = -1
            elif points == self.points + 1:
                r = 1
            else:
                r = -0.1
            # update its Q-value estimate for the state-action pair Q(s_t, a_t)
            temp = - math.inf
            for i in range(3, -1, -1):
                q = self.Q[s_prime][i]
                if q > temp:
                    temp = q
            self.N[self.s][self.a] += 1
            self.Q[self.s][self.a] += (self.C / (self.N[self.s][self.a] + self.C)) * (r + self.gamma * temp - self.Q[self.s][self.a])
        if dead:
            self.reset()
            return 3
        
        # agent is now in state s_{t+1} and choosing the Optimal Action
        self.s = s_prime
        self.points = points
        temp = - math.inf
        for i in range(3, -1, -1):
            if self.N[s_prime][i] < self.Ne:
                f = 1
            else:
                f = self.Q[s_prime][i]
            if f > temp:
                temp = f
                self.a = i

        return self.a


    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        cur_state = [0, 0, 0, 0, 0, 0, 0, 0]

        # food_dir_x
        if environment[0] > environment[3]:
            cur_state[0] = 1
        if environment[0] < environment[3]:
            cur_state[0] = 2
        # food_dir_y   
        if environment[1] > environment[4]:
            cur_state[1] = 1
        if environment[1] < environment[4]:
            cur_state[1] = 2

        # adjoining_wall_x
        if environment[0] < 2 and environment[0] > 0:
            cur_state[2] = 1
        if environment[0] > utils.DISPLAY_WIDTH - 3 and environment[0] < utils.DISPLAY_WIDTH - 1:
            cur_state[2] = 2
        # adjoining_wall_y
        if environment[1] < 2 and environment[1] > 0:
            cur_state[3] = 1
        if environment[1] > utils.DISPLAY_HEIGHT - 3 and environment[1] < utils.DISPLAY_HEIGHT - 1:
            cur_state[3] = 2

        # adjoining_body
        if (environment[0], environment[1] - 1) in environment[2]:
            cur_state[4] = 1
        if (environment[0], environment[1] + 1) in environment[2]:
            cur_state[5] = 1
        if (environment[0] - 1, environment[1]) in environment[2]:
            cur_state[6] = 1
        if (environment[0] + 1, environment[1]) in environment[2]:
            cur_state[7] = 1

        return tuple(cur_state)