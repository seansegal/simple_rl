#!/usr/bin/env python

# Python imports.
import random
import sys

# Other imports.
import srl_example_setup
from simple_rl.tasks import GatheringMDP
from simple_rl.run_experiments import play_markov_game
from simple_rl.agents import FixedPolicyAgent

from simple_rl.agents.dqn.DQNAgentClass import DQNAgent
from simple_rl.tasks import RockPaperScissorsMDP

def main(open_plot=True):
    # gamma, N_apples, N_tagged = [0.99, 1, 1]
    # possible_apple_locations = [(17,4), (16,5), (17,5), (18,5), (15,6), (16,6), (17,6), (18,6), (19,6), (16,7), (17,7), (18,7), (17,8)]
    # gathering = GatheringMDP(gamma, possible_apple_locations, N_apples, N_tagged)
    # gathering.get_init_state().show()

    markov_game = RockPaperScissorsMDP()
    dqn = DQNAgent(markov_game.get_actions(), x_dim=1, y_dim=1, eps_start=1.0, eps_end=0.1, num_channels=1)
    fixed_action = random.choice(markov_game.get_actions())
    fixed_agent = FixedPolicyAgent(policy=lambda s:fixed_action)
    play_markov_game([dqn, fixed_agent], markov_game, instances=15, episodes=100, steps=100, open_plot=open_plot)

    # play_markov_game([AGENTS_HERE], gathering, instances=1, episodes=100000, steps=1000, verbose=False, open_plot=True)


if __name__ == "__main__":
    main(open_plot=not(sys.argv[-1] == "no_plot"))
