import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import layers, Model
import os

print(device_lib.list_local_devices())
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_gpu_available())

GAME = 'flappy_bird_q_learning'
ACTIONS = 2
GAMMA = 0.99

OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1

REPLAY_MEMORY = 25000
BATCH = 32
FRAME_PER_ACTION = 1

class QNetwork(tf.keras.Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        # This layer now uses sigmoid activation to output a probability
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # The output will be a single scalar between 0 and 1
        x = self.fc2(x)
        return x
    
optimizer = tf.keras.optimizers.Adam(1e-6)
loss_function = tf.keras.losses.BinaryCrossentropy()

def train_step(model, minibatch):
    with tf.GradientTape() as tape:
        states = np.array([d[0] for d in minibatch], dtype=np.float32)
        actions = np.array([d[1] for d in minibatch], dtype=np.int32)
        rewards = np.array([d[2] for d in minibatch], dtype=np.float32)
        next_states = np.array([d[3] for d in minibatch], dtype=np.float32)
        done = np.array([d[4] for d in minibatch], dtype=np.float32)

        # predict Q-values for starting state and next states
        q_values = model(states)
        q_values_next = model(next_states)
        y_batch = tf.where(done, rewards, rewards + GAMMA * tf.reduce_max(q_values_next, axis=1))
       
        q_action = tf.reduce_sum(tf.multiply(q_values, actions), axis=1)
        loss = tf.reduce_mean(tf.square(y_batch - q_action))

        # vackpropagation
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return q_values, loss

latest_model_path = None

try:
    saved_models_dir = 'saved_networks'
    model_prefix = 'model_'
    model_files = os.listdir(saved_models_dir)
    model_files = [f for f in model_files if f.startswith(model_prefix) and f[len(model_prefix):].isdigit()]
    latest_model_file = sorted(model_files, key=lambda x: int(x[len(model_prefix):]), reverse=True)[0]
    latest_model_path = os.path.join(saved_models_dir, latest_model_file)
except Exception as e:
    print(e)

def network_init(model):
    model.compile(optimizer=optimizer, loss=loss_function)

    game_state = game.GameState()
    D = deque()

    q_file = open("logs_" + GAME + "/q_values.txt", 'w')
    l_file = open("logs_" + GAME + "/loss.txt", 'w')
    s_file = open("logs_" + GAME + "/std_out.txt", 'w')

    # Get the first state by doing nothing and preprocess the image to 80x80x4
    zero_action = np.zeros(ACTIONS)
    zero_action[0] = 1
    x_t, r_0, terminal = game_state.frame_step(zero_action)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    if(latest_model_path):
        try:
            model = tf.keras.models.load_model(latest_model_path)
            print(f"Successfully loaded the model from {latest_model_path}")
        except Exception as e:
            print("Could not load the latest model. Error:", e)

    # Start training
    epsilon = INITIAL_EPSILON
    t = 0
    # infinity and beyond
    while "flappy" != "bird":
        # epsilon-greedily
        prd = model.predict(np.array([s_t], dtype=np.float32))
        readout_t = prd[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("Random Initiated")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # zero state

        # scale epsilon down
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # selected action and observe the next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # train if done observing
        if t > OBSERVE:
            # minibatch to train on
            minibatch = random.sample(D, BATCH)
            q_values, loss = train_step(model, minibatch)
            l_file.write(str(loss.numpy()) + '\n')
            q_file.write(",".join([str(x) for x in q_values[0].numpy()]) + '\n')
        s_t = s_t1
        t += 1

        # progress every 10000 iterations
        if t % 10000 == 0:
            model.save('saved_networks/model_'+ str(t))
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        info_string = "FRAME {},STATE {},EPSILON {},ACTION {},REWARD {},Q_MAX {:e}".format(
            t, state, epsilon, action_index, r_t, np.max(readout_t)
        )

        # Now you can print this string
        print(info_string)
        s_file.write(info_string + "\n")
        
def init_game():
    model = QNetwork()
    network_init(model)

def main():
    init_game()

if __name__ == "__main__":
    main()

