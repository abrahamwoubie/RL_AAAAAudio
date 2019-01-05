from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from GlobalVariables import GlobalVariables
from keras.layers import Dense, Conv2D, Flatten,Conv1D, MaxPooling2D,Convolution2D,GlobalAveragePooling2D
from keras import optimizers
import random
import numpy as np
from keras.layers import MaxPooling1D,GlobalAveragePooling1D,Dropout,LSTM,TimeDistributed,AveragePooling1D,Embedding,Activation

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parameter=GlobalVariables
grid_size=GlobalVariables
options=GlobalVariables

class DQNAgent:
    def __init__(self,env):
        if(options.use_samples):
            self.state_dim=parameter.sample_state_size
        elif(options.use_pitch):
            self.state_dim = parameter.pitch_state_size
        elif(options.use_spectrogram):
            self.state_dim = parameter.spectrogram_state_size
        else:
            self.state_dim = parameter.raw_data_state_size
        self.action_dim=parameter.action_size
        self.memory = deque(maxlen=2000)
        self.discount_factor = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        if(options.use_pitch or options.use_samples):
            self.model = self._build_model()
        else:
            self.model=self._build_CNN_model_2D()
        print(self.model.summary())

    def _build_model(self):
        
        print("Neural Net for Deep-Q learning Model,Dense Network")
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_dim,), kernel_initializer='uniform', activation='relu'))
        model.add(Dense(24, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
        return model


    def _build_CNN_model_2D(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                         input_shape=(parameter.spectrogram_length, parameter.spectrogram_state_size, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model

    # model = Sequential()
        # model.add(Conv1D(100, 10, activation='relu', input_shape=(self.state_dim,1)))
        # model.add(Conv1D(100, 10, activation='relu'))
        # model.add(MaxPooling1D(3))
        # model.add(Conv1D(160, 10, activation='relu'))
        # model.add(Conv1D(160, 10, activation='relu'))
        # model.add(MaxPooling1D(3))
        # model.add(Dropout(0.5))
        # model.add(Dense(parameter.action_size, activation='softmax'))
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss='categorical_crossentropy', optimizer='sgd')
        # return model

        # model = Sequential()
        # model.add(Conv1D(100, 10, activation='relu', input_shape=(self.state_dim,1)))
        # model.add(Conv1D(100, 10, activation='relu'))
        # model.add(MaxPooling1D(3))
        # model.add(Conv1D(160, 10, activation='relu'))
        # model.add(Conv1D(160, 10, activation='relu'))
        # model.add(MaxPooling1D(3))
        # model.add(Dropout(0.5))
        # model.add(Dense(parameter.action_size, activation='softmax'))
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss='categorical_crossentropy', optimizer='sgd')
        # return model

    def replay_memory(self, state, action, reward, next_state, done):
        #print(state, action, reward, next_state, done)
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def act(self, state,env):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(env.allowed_actions())
        else:
            act_values = self.model.predict(state)
            state = env.state;
            actions_allowed = env.allowed_actions()
            actions_allowed.sort()
            q_value_allowed = []
            for i in range(parameter.action_size):
                if i in actions_allowed:
                    q_value_allowed.append(act_values[0][i])
                else:
                    q_value_allowed.append(-100)
            #print("Action Values",act_values)
            #print("Allowed Actions at state {} are {}".format(state,actions_allowed))
            #print("q Values allowed", q_value_allowed)
            return  np.argmax(q_value_allowed)  # returns action


    # def act(self, state,env):
    #     if np.random.rand() <= self.epsilon:
    #         # The agent acts randomly
    #         return np.random.choice(env.allowed_actions())
    #     else:
    #         actions_allowed = env.allowed_actions()
    #         #print("actions allowed",actions_allowed)
    #         # Predict the reward value based on the given state
    #         state = np.float32(state)
    #         q_values = self.model.predict(state)
    #         #print("q_values",q_values)
    #         return np.argmax(q_values)

    # def act(self, state,env):
    #     #if random.uniform(0, 1) < self.epsilon:
    #     return np.random.choice(env.allowed_actions())


    # def act(self,state,env):
    #     if random.uniform(0, 1) < self.epsilon:
    #         # explore
    #         return np.random.choice(env.allowed_actions())
    #     else:
    #         act_values = self.model.predict(state)
    #         state = env.state;
    #         actions_allowed = env.allowed_actions()
    #         actions_allowed.sort()
    #         q_value_allowed = []
    #
    #         for i in range(parameter.action_size):
    #             if i in actions_allowed:
    #                 q_value_allowed.append(act_values[0][i])
    #         action = np.argmax(q_value_allowed)
    #         print("actions allowed{} for state{}".format(actions_allowed,state))
    #         print("act values",q_value_allowed)
    #         return action


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
                #target = (reward + self.discount_factor * np.max(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
