from Environment import  Environment
from AgentQL import Agent
import matplotlib.pyplot as plt
from GlobalVariables import  GlobalVariables
from ExtractFeatures import Extract_Features
import pylab

grid_size=GlobalVariables
parameter=GlobalVariables

env = Environment(grid_size.nRow,grid_size.nCol)
agent = Agent(env)

sample=Extract_Features

import numpy as np
# Train agent
print("\nTraining agent...\n")

Number_of_Iterations=[]
Number_of_Episodes=[]
list=[]


for i in range(parameter.how_many_times):

    for episode in range(parameter.Number_of_episodes):

        # Generate an episode
        iter_episode, reward_episode = 0, 0
        state,goal_state,wall = env.reset()  # starting state
        Number_of_Episodes.append(episode)
        iteration=0
        #while True:
        for i in range(parameter.timesteps):
            iteration+=1
            action = agent.get_action(env)  # get action

            state_next, reward, done = env.step(action)#,goal_state)  # evolve state by action
            agent.train((state, action, state_next, reward, done))  # train agent

            #state_sample, reward, done = env.step(action)  # evolve state by action
            #agent.train((state, action, state_sample, reward, done))  # train agent

            iter_episode += 1
            reward_episode += reward
            if done:
                break
            state = state_next  # transition to next state
            #state = state_sample

        Number_of_Iterations.append(iteration)
        # Decay agent exploration parameter
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

        # Print
        # if (episode == 0) or (episode + 1) % 10 == 0:
        print("[episode {}/{}], iter = {}, reward = {:.1F}".format(
            episode + 1, parameter.Number_of_episodes, iter_episode, reward_episode))

    list.append(Number_of_Iterations)

    fig = plt.figure()
    fig.suptitle('Q-Learning', fontsize=12)
    title=str(grid_size.nRow) + "X" + str(grid_size.nCol) + '_'+ str(i+1)
    fig.suptitle(title, fontsize=12)
    plt.plot(np.arange(len(Number_of_Episodes)), Number_of_Iterations)
    plt.ylabel('Number of Iterations')
    plt.xlabel('Episode Number')
    #filename=title+'.png'
    #plt.savefig(filename)
    plt.show(block=False)
#    plt.pause(3)
    plt.close()

mu=np.mean(list, axis=0)
std=np.std(list, axis=0)

Episode_Number = []
for i in range(1,len(list[0])+1):
    Episode_Number.append(i)

pylab.plot(Episode_Number, mu, '-b', label='Mean')
pylab.plot(Episode_Number, std, '-r', label='Standard Deviation')
pylab.legend(loc='upper left')
pylab.ylim(0, max(np.max(mu),np.max(std))+1)
pylab.xlim(1, np.max(Episode_Number)+1)
pylab.xlabel('Episode Number')
pylab.ylabel('Iteration')
pylab.savefig('Test.png')
pylab.show()
