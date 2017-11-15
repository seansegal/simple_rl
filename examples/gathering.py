#!/usr/bin/env python

# Python imports.
import random
import sys

# Other imports.
import srl_example_setup
from simple_rl.tasks import GatheringMDP
from simple_rl.run_experiments import play_markov_game

def main(open_plot=True):
    gamma, N_apples, N_tagged = [0.99, 1, 1]
    possible_apple_locations = [(17,4), (16,5), (17,5), (18,5), (15,6), (16,6), (17,6), (18,6), (19,6), (16,7), (17,7), (18,7), (17,8)]
    gathering = GatheringMDP(gamma, possible_apple_locations, N_apples, N_tagged)
    gathering.get_init_state().show()


if __name__ == "__main__":
    main(open_plot=not(sys.argv[-1] == "no_plot"))
