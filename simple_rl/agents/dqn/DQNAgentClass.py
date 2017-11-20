''' DQNAgentClass.py: Class for Deep Q-network agent. Built based on the network
in DeepMind, Multi-agent RL in Sequential Social Dilemmas paper. '''

# Python imports.
from simple_rl.agents.AgentClass import Agent
import tensorflow as tf

class DQNAgentClass(Agent):

    NAME = "dqn-deep-mind"

    def __init__(self, name=NAME, learning_rate=1e-4, num_actions=8, x_dim=21, y_dim=16, eps_start=1.0, eps_end=0.1):
        Agent.__init__(self, name=name, actions=[])
        self.learning_rate = epsilon, learning_rate
        self.hidden_layers = [32, 32]
        self.mainQN = QNetwork()
        self.targetQN = QNetwork()
        self.sess = tf.Session()
        self.experience_buffer = ExperienceBuffer(buffer_size=10e4)
        self.prev_state, self.prev_action = None, None
        self.epsilon, self.epsilon_decay, self.epsilon_end = eps_start, 0.0001, eps_end
        self.curr_step, self.total_steps = 0, 0
        self.curr_episode = 0
        self.update_freq = 4
        self.batch_size = 32
        self.should_train = True

    def act(self, state, reward):
        # TODO: Everything! (Batching, replay, run graph)
        # TODO: choose action with epsilon greedy algorithm

        if self.should_train and self.total_steps % self.update_freq == 0:
            sample = self.experience_buffer.sample(self.batch_size)
            # TODO: actually pass the image below 
            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.image: np.vstack(trainBatch[:,3])})
            Q2 = sess.run(targetQN.out, feed_dict={targetQN.image: np.vstack(trainBatch[:,3])})
            # TODO: train
            pass

        # If not training
        if random.random() < self.epsilon:
            action =  np.random.choice(self.num_actions) # NOTE:  Again assumes actions encoded 0...7
        else:
            img = state.to_rbg()
            action = mainQN.get_best_action(self.sess, img)

        if self.prev_state != None and self.prev_action != None:
            self.experience_buffer.add((self.prev_state, self.prev_action, reward, state.to_rbg(), state.is_terminal()))

        self.prev_state, self.prev_action = state.to_rbg(), action

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

        self.curr_step += 1
        self.total_steps += 1
        if state.is_terminal():
            self.curr_step = 0
            self.curr_episode += 1

        return action


    def __str__(self):
        return str(self.name)


class QNetwork():
    def __init__(self, learning_rate=1e-4, num_actions=8, x_dim=21, y_dim=16):
        self.hidden_layers = [32, 32]
        self.num_actions = num_actions
        self.x_dim, self.y_dim = x_dim, y_dim
        self.num_channels = 3


        self.image = tf.placeholder(tf.float32, shape=[None, self.num_channels, self.x_dim, self.y_dim])
        self.targetQ = tf.placeholder(tf.float32, shape=[None])

        # NOTE: Assumes that actions are ordered 0...7. TODO: check if true
        self.actions = tf.placeholder(tf.int32, shape=[None])

        self.out = setup_network(self.image)

        self.loss_val = self.loss(self.out)

        self.train_op = self.train(self.loss_val)

    def setup_network(inpt):
        flattened_input = tf.reshape(inpt, [-1, self.num_channels*self.x_dim*self.y_dim])

        curr_layer = flattened_input
        for i, layer_size in enumerate(self.hidden_layers):
            curr_layer = tf.layers.dense(curr_layer, units=layer_size, activation=tf.nn.relu)

        return tf.layers.dense(curr_layer, units=self.num_actions)

    def loss(output):
        actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
        Q = tf.reduce_sum(actions_onehot * self.out, axis=1)
        return tf.reduce_mean(tf.square(self.targetQ - Q))

    def train(loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def get_best_action(self, sess, img):
        self.predict = tf.argmax(self.out, 1)
        return sess.run(self.predict, feed_dict={self.image: img})


class ExperienceBuffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) == buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, size):
        indexes = np.random.randint(0, high=len(self.buffer), size=size)
        samples = [0] * size
        for i, index in enumerate(indexes):
            samples[i] = self.buffer[index]
        return samples

if __name__ == '__main__':
    print 'HERE'
