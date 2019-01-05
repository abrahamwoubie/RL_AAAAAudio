import numpy as np
import os, sys, random, operator
from scipy.spatial import  distance
from GlobalVariables import GlobalVariables
from ExtractFeatures import Extract_Features
from GlobalVariables import GlobalVariables

Extract=Extract_Features
grid_size=GlobalVariables
options=GlobalVariables

class Environment:

    def __init__(self, Ny, Nx):
        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size
        self.state_dim = (Ny, Nx)
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (0, 0)
        self.goal_state=(grid_size.nRow-1,grid_size.nCol-1)
        self.wall=(1,1)
        #
        # start_row = random.choice(range(0, grid_size.nRow - 1))
        # start_col = random.choice(range(0, grid_size.nCol - 1))
        # #
        # goal_row = random.choice(range(0, grid_size.nRow - 1))
        # goal_col = random.choice(range(0, grid_size.nCol - 1))
        # #
        #
        # wall_row=random.choice(range(0,grid_size.nRow-1))
        # wall_col=random.choice(range(0,grid_size.nCol-1))
        #
        # while(wall_row==start_row and wall_col==start_col):
        #     wall_row = random.choice(range(0, grid_size.nRow - 1))
        #     wall_col = random.choice(range(0, grid_size.nCol - 1))
        #
        # while(wall_row!=goal_row and wall_col!=goal_col):
        #     wall_row = random.choice(range(0, grid_size.nRow - 1))
        #     wall_col = random.choice(range(0, grid_size.nCol - 1))

        #
        # self.state = (start_row, start_col)
        # self.goal_state=(goal_row,goal_col)
        # self.wall = (wall_row,wall_col)

        return self.state, self.goal_state,self.wall

    def step(self, action,samples_goal):

        reward = 0
        done = False
        if(action==0): # up
            state_next =  (self.state[0]-1) , self.state[1]

        if(action==1): #right
            state_next = self.state[0] , (self.state[1] + 1)

        if(action==2): # down
            state_next = (self.state[0] + 1) , self.state[1]

        if(action==3): # left
            state_next = self.state[0]  , (self.state[1] - 1)

        # if(state_next[0]<0 or state_next[1]<0):
        #     state_next=self.state
        # if(state_next[0]>grid_size.nRow-1  or state_next[1]>grid_size.nRow-1 ):
        #     state_next = self.state
        # if(state_next[0]==grid_size.nRow-1 and state_next[1]==grid_size.nCol-1):
        #     reward=1
        #     done= True
        #print(action,state_next)
        #
        # if(state_next[0]==self.wall[0] and state_next[1]==self.wall[1]): # If next state is a wall, it can't pass
        #     #print("Can pass through the wall")
        #     if(options.use_samples):
        #         samples=Extract.Extract_Samples(self.state[0],self.state[1])
        #     elif(options.use_pitch):
        #         samples = Extract.Extract_Pitch(self.state[0], self.state[1])
        #     elif(options.use_spectrogram):
        #         samples = Extract.Extract_Spectrogram(self.state[0], self.state[1])
        #     else:
        #         samples = Extract.Extract_Raw_Data(self.state[0], self.state[1])
        #     return samples,reward,done
        # else:

        if (options.use_samples):
            samples_current=Extract.Extract_Samples(state_next[0],state_next[1])
        if (options.use_pitch):
            samples_current = Extract.Extract_Pitch(state_next[0], state_next[1])
        if (options.use_spectrogram):
            samples_current = Extract.Extract_Spectrogram(state_next[0], state_next[1])

        '''
        if(options.use_samples):
            if (state_next[0] == self.goal_state[0] and state_next[1] == self.goal_state[1]):
            #if (distance.euclidean(samples_goal, samples_current) == 0):
                reward = 1
                done = True
        if (options.use_pitch):
            #if (distance.euclidean(samples_goal, samples_current) == 0):
            if (state_next[0] == self.goal_state[0] and state_next[1] == self.goal_state[1]):
                reward = 1
                done = True
        if(options.use_spectrogram):
            if (np.mean(samples_goal)==np.mean(samples_current)):
                reward = 1
                done = True
        '''

        if(state_next[0] == self.goal_state[0] and state_next[1] == self.goal_state[1]):
                reward = 1
                done = True

        # Update state
        self.state = state_next
        return samples_current, reward, done
        #return state_next,reward,done

    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        if (y > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed
