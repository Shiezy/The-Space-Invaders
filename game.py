import gym
from gym.wrappers import Monitor
from gym import logger as gymlogger
import tensorflow as tf
import numpy as np
import random

import math
import glob
import io
import base64
from IPython.display import HTML

from IPython import display as ipythondisplay

import keras
from keras.callbacks import TensorBoard
import os.path
import time
import shutil
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import load_model
from collections import deque 
from datetime import datetime
from keras.models import clone_model
from skimage.color import rgb2gray
from skimage.transform import resize
import time

observe_step_num = 10000 # The frequency at which the target network is updated
batch_size = 32 # Number of training cases that are computed in each update
gamma = 0.99 # Discount factor
replay_memory = 350000 # This many recent frames are used for sampling
num_episode = 100000 # The number of episodes
learning_rate = 0.00025 # The learning rate
init_epsilon = 1.0 # Initial value of epsilon in epsilon-greedy method
final_epsilon = 0.1 # Final value of epsilon in epslion-greedy method
epsilon_step_num = 1000000 # Number of frames to get from the initial to final value of epsilon
refresh_target_model_num = 10000 # The frequency at which the target model updates
no_op_steps = 30 # Max number of "do nothing" steps at the beginning of an episode
train_dir = "training" # Training directory
tensorboard_dir = "logs" # Tensorboard directory
restore_file_path = "/home/sheila/Desktop/Learning data science/Space-Invaders-master/training/SpaceInvaders_20190320110701.h5" # Restore file path
model_restore_dir = "/home/sheila/Desktop/Learning data science/Space-Invaders-master/training/SpaceInvaders_20190320110701.h5" # Model file path
num_test_episodes = 1 # Number of episodes you test for
resume = False


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84 , 84), mode='constant') * 255)
    return processed_observe

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

ATARI_SHAPE = (84, 84, 4) # 84 by 84 and 4-stack to help Neural Net determine direction
ACTION_SIZE = 4

def atari_model():
    
    # Input layers.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((ACTION_SIZE,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input) # Lambda is used in Keras to perform math
    
    # "The first hidden layer convolves 32 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.Conv2D(
        32, (8, 8), strides=4, activation='relu', name="Conv1"
    )(normalized)
    
    # "The second hidden layer convolves 64 4×4 filters with stride 2, followed by a rectifier nonlinearity."
    conv_2 = keras.layers.Conv2D(
        64, (4, 4), strides=2, activation='relu', name="Conv2"
    )(conv_1)
    
    # "The third hidden layer convolves 64 3x3 filters with stride 1, again followed by a rectifier nonlinearity."
    conv_3 = keras.layers.Conv2D(
        64, (3, 3), strides=1, activation='relu', name="Conv3"
    )(conv_2)
    
    # Flattening the convolutional layer.
    conv_flattened = keras.layers.core.Flatten(name="flatten")(conv_3) # Taking an array and converting it into a linear vector
    
    # "Fully connected layer made up of 512 rectifier units."
    hidden = keras.layers.Dense(512, activation='relu', name="Dense512")(conv_flattened)
    
    # The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(ACTION_SIZE, name="Output")(hidden)
    
    # Finally, we multiply the output by the mask.
    filtered_output = keras.layers.Multiply(name='QValue')([output, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    model.summary()
    return model

# get action from model using epsilon-greedy policy
def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= observe_step_num:
        return random.randrange(ACTION_SIZE)
    else:
        q_value = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        return np.argmax(q_value[0])


# save sample <s,a,r,s'> to the replay memory
def store_memory(memory, history, action, reward, next_history, dead):
    memory.append((history, action, reward, next_history, dead))

# get one hot
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]
  
# train model by taking a random batch from memory
def train_memory_batch(memory, model_target, model):
    mini_batch = random.sample(memory, batch_size)
    
    history = np.zeros((batch_size, ATARI_SHAPE[0],
                        ATARI_SHAPE[1], ATARI_SHAPE[2]))
    next_history = np.zeros((batch_size, ATARI_SHAPE[0],
                             ATARI_SHAPE[1], ATARI_SHAPE[2]))
    target = np.zeros((batch_size,))
    action, reward, dead = [], [], []

    for idx, val in enumerate(mini_batch):
        history[idx] = val[0]
        next_history[idx] = val[3]
        action.append(val[1])
        reward.append(val[2])
        dead.append(val[4])

    actions_mask = np.ones((batch_size, ACTION_SIZE))
    next_Q_values = model_target.predict([next_history, actions_mask])

    # like Q Learning, get maximum Q value at s'
    # But from target model
    for i in range(batch_size):
        if dead[i]:
            target[i] = -1
        else:
            target[i] = reward[i] + gamma * np.amax(next_Q_values[i])

    action_one_hot = get_one_hot(action, ACTION_SIZE)
    target_one_hot = action_one_hot * target[:, None]

    h = model.fit(
        [history, action_one_hot], target_one_hot, epochs=1,
        batch_size=batch_size, verbose=0)

    return h.history['loss'][0]
  
  
# Display Video
  
def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

def train():
    env = gym.make("SpaceInvaders-v0")

    # deque: Once a bounded length deque is full, when new items are added,
    # a corresponding number of items are discarded from the opposite end
    memory = deque(maxlen=replay_memory)
    episode_number = 0
    epsilon = init_epsilon
    epsilon_decay = (init_epsilon - final_epsilon) / epsilon_step_num
    global_step = 0

    if resume:
        model = load_model(restore_file_path, custom_objects={'huber_loss': huber_loss})
        epsilon = 0.1
    else:
        model = atari_model()

    # initialize file writer for tensorboard
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}-log".format(tensorboard_dir, now)
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    # clone model
    model_target = clone_model(model)
    model_target.set_weights(model.get_weights())

    while episode_number < num_episode:

        done = False
        dead = False
        # 1 episode = 3 lives
        step, score, start_life = 0, 0, 3
        loss = 0.0
        observe = env.reset()

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, no_op_steps)):
            observe, _, _, _ = env.step(1)
        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:

            # get action for the current history and go one step in environment
            action = get_action(history, epsilon, global_step, model)

            # scale down epsilon, the epsilon only begin to decrease after observe steps
            if epsilon > final_epsilon and global_step > observe_step_num:
                epsilon -= epsilon_decay

            observe, reward, done, info = env.step(action)
            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)
            
            if start_life > info['ale.lives']:
              dead = True
              start_life = info['ale.lives']

            # save the statue to memory, each replay takes 2 * (84*84*4) bytes = 56448 B = 55.125 KB
            store_memory(memory, history, action, reward, next_history, dead)  #

            # check if the memory is ready for training
            if global_step > observe_step_num:
                loss = loss + train_memory_batch(memory, model_target, model)
                
                if global_step % refresh_target_model_num == 0:  # update the target model
                    model_target.set_weights(model.get_weights())
                    print ("Target Model Refreshed")

            score += reward

            history = next_history

            global_step += 1
            step += 1

            if done:
                if global_step <= observe_step_num:
                    state = "observe"
                elif observe_step_num < global_step <= observe_step_num + epsilon_step_num:
                    state = "explore"
                else:
                    state = "train"
                print('state: {}, episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}'
                      .format(state, episode_number, score, global_step, loss / float(step), step, len(memory)))

                if episode_number % 15 == 0 or (episode_number + 1) == num_episode:
                #if episode_number % 1 == 0 or (episode_number + 1) == num_episode:  # debug
                    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    file_name = "SpaceInvaders_{}.h5".format(now)
                    model_path = os.path.join(train_dir, file_name)
                    model.save(model_path)
                    # shutil.copy("/content/{}".format(model_path),"/content/gdrive/My Drive/Colab Notebooks/Atari Replicate")

                # Add user custom data to TensorBoard
                loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="loss", simple_value=loss / float(step))])
                file_writer.add_summary(loss_summary, global_step=episode_number)

                score_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="score", simple_value=score)])
                file_writer.add_summary(score_summary, global_step=episode_number)

                episode_number += 1

    file_writer.close()
    
# train()

def test():
    env = wrap_env(gym.make('SpaceInvaders-v0'))

    episode_number = 0
    epsilon = 0.001
    global_step = 0
    model = load_model(model_restore_dir, custom_objects={'huber_loss': huber_loss})  # load model with customized loss func

    while episode_number < num_test_episodes:
        
        done = False
        dead = False
        
        # 1 episode = 5 lives
        score, start_life = 0, 3
        observe = env.reset()

        observe, _, _, _ = env.step(1)
        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        
        while not done:
            
            # get action for the current history and go one step in environment
            q_value = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
            action = np.argmax(q_value[0])

            # step in environment
            observe, reward, done, info = env.step(action)
            
            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
              dead = True
              start_life = info['ale.lives']

            # TODO: may be we should give negative reward if miss ball (dead)
            score += reward
            
            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky
            if dead:
              dead = False
              print ("agent is dead")
              history = next_history
            else:
              history = next_history
              
            # print("step: ", global_step)
            global_step += 1

            if done:
                episode_number += 1
                print('episode: {}, score: {}'.format(episode_number, score))
                
    env.close()
    show_video()
test()