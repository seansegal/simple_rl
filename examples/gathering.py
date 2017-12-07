#!/usr/bin/env python

# Python imports.
import random
import sys
from simple_rl.agents import RandomAgent, FixedPolicyAgent

# Other imports.
import srl_example_setup
from simple_rl.tasks.gathering.GatheringMDPClass import GatheringMDP
from simple_rl.run_experiments import play_markov_game
from simple_rl.agents import FixedPolicyAgent

from simple_rl.agents.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks import RockPaperScissorsMDP

def main(open_plot=True):
    gamma, N_apples, N_tagged = [0.99, 1, 1]
    possible_apple_locations = [(17,4), (16,5), (17,5), (18,5), (15,6), (16,6), (17,6), (18,6), (19,6), (16,7), (17,7), (18,7), (17,8)]
    gathering = GatheringMDP(gamma, possible_apple_locations, N_apples, N_tagged)

    rand_agent = RandomAgent(actions=gathering.get_actions())

    dqn = DQNAgent(gathering.get_actions(), num_channels=3)
    # fixed_action = random.choice(gathering.get_actions())
    fixed_agent = FixedPolicyAgent(policy=lambda s:"step_left")
    play_markov_game([dqn, rand_agent], gathering, instances=1, episodes=1, steps=10000, open_plot=open_plot)
    # augment play_markov_game with a named parameter with default false
    # gathering.get_init_state().show()


if __name__ == "__main__":
    main(open_plot=not(sys.argv[-1] == "no_plot"))
