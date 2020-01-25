"""Reinforcement Learning with OpenAI gym.

ref:
    - gym.openai.com
"""
import numpy as np
import gym
import tensorflow as tf

"""Concept 0: CartPole

Example: CartPole Environment
    - goal: to balance the pole on the cart
    - actions: move left or right the pole
    - environment objects:
        observation
            - horizontal position
            - horizontal velocity
            - angle of pole
            - angular velocity
        reward
            - reward by previous action
            - varied, but normally, increase
        done
            - reset environement
        info

"""

"""Concept 1: Create an environment
Policy 0:
    randomly action
"""

# create environment of cartpole (pre-defined)
env = gym.make('CartPole-v0')

# check observation space -> eg. here 4 float number
print(env.observation_space)

print("Initial env")
observation = env.reset()
print(observation)


steps = 10
for _ in range(steps):
    env.render()

    # randomly action of ~ Left (0) and Right (1)
    print(env.action_space)
    action = env.action_space.sample()
    print(action)

    observation, reward, done, info = env.step(action)

    print(info)


"""Concept 2: Policy of actions

Policy 1:
    move cart to right -> if the pole false to the right
    move cart to left -> vice versa
"""

# Create env
env = gym.make('CartPole-v0')

# Init
observation = env.reset()
steps = 10

for _ in range(steps):
    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    # move right -> if false to right -> else move left
    if pole_ang > 0:
        action = 1
    else:
        action = 0

    observation, reward, done, info = env.step(action)


"""Concept 3: Simple NN + Policy of actions

NN
    Input: observation array (4 numbers)
    Output: 2 probabilites [left, right]

Policy 2:
    random action based on prob.

Summary:
    NN  -> prob
            -> sampling based on prob weights
                -> action (move left~0 or right~1)
Notice:
    - not choose highest prob to move
        -> could be "infinitive loop"

    - but randomly based on prob-weights
"""
num_inputs = 4
num_hidden = 4
num_outputs = 1  # prob to go left, 1 - p = prob to go light

# ? You could ask why using AutoEncorder here?
initializer = tf.contrib.layers.variance_scaling_initializer()

# Place holder
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer_one = tf.layers.dense(X, num_hidden,
                                   activation="relu",
                                   kernel_initializer=initializer)

hidden_layer_two = tf.layers.dense(hidden_layer_one, num_hidden,
                                   activation="relu",
                                   kernel_initializer=initializer)

output_layer = tf.layers.dense(hidden_layer_two, num_outputs,
                               activation="sigmoid",
                               kernel_initializer=initializer)
probabilites = tf.concat(axis=1, values=[output_layer, 1 - output_layer])

# Randomly sampling given p (here binomial)
action = tf.multinomial(probabilites, num_samples=1)

init = tf.global_variables_initializer()
step_limits = 500

env = gym.make("CartPole-v0")
episodes = 50
avg_steps = []
with tf.Session() as sess:
    sess.run(init)

    for i in range(episodes):
        obs = env.reset()

        # feed init value of obs
        feed_dict = {X: obs.reshape(1, num_inputs)}
        for step in range(step_limits):

            # ? What is cost function here?
            #   seemly feed forward and no update weights
            #   purpose to get the p value of left?
            action_val = action.eval(feed_dict=feed_dict)
            _act = action_val[0][0]   # return 0 or 1

            obs, reward, done, info = env.step(_act)

            if done:
                avg_steps.append(step)
                print(f"Done after {step} steps")
                break
mean_step = np.mean(avg_steps)
print(f"After {episodes} Episodes, average steps per game was {mean_step}")
env.close()

# Result here could be worse, ~ 20 steps
#   even we increase episodes
#   -> because of choosing action policy (binomial given p)
#       -> may choose a wrong one
#   -> we need a better policy -> policy gradient theory
