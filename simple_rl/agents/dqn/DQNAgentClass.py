''' DQNAgentClass.py: Class for Deep Q-network agent. Built based on the network
in DeepMind, Multi-agent RL in Sequential Social Dilemmas paper. '''

# Python imports.
from simple_rl.agents.AgentClass import Agent

class DQNAgentClass(Agent):

    NAME = "dqn-deep-mind"

    def __init__(self, name=NAME):
        Agent.__init__(self, name=name, actions=[])
        # TODO: Initialize parameters: epsilon, layer sizes, batch sizes...etc
        # TODO: Setup Tensorflow Graph.



    def act(self, state, reward):
        # TODO: Everything! (Batching, replay, run graph)
        # TODO: choose action with epsilon greedy algorithm
        pass


    def __str__(self):
        return str(self.name)
