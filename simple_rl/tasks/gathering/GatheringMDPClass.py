''' GatheringDilemmaMDPClass.py: Contains an implementation of Gathering from
the Deep Mind paper Multi-agent Reinforcement Learning in Sequential Social
Dilemmas. '''

# Python imports.
import random

# Other imports.
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from simple_rl.tasks.gathering.GatheringStateClass import GatheringState

# TODO: remove when State is changed to GatheringState
from simple_rl.mdp.StateClass import State

class GatheringMDP(MarkovGameMDP):

    # Static constants.
    ACTIONS = ["step_forward", "step_backward", "step_left", "step_right", "rotate_left", "rotate_right", "use_beam", "stand_still"]

    #ACTIONS = ["rock", "paper", "scissors"]

    def __init__(self, gamma, N_apples, N_freeze):
        # TODO: change State to GatheringState()
        MarkovGameMDP.__init__(self, GatheringMDP.ACTIONS, self._transition_func, self._reward_func, init_state=State())

        # TODO: 1. Take in game parameters: grid game size (16 by 21 in the paper),
        # game length, player 1 & player 2 locations, gamma, n_{apples}, n_{freeze}
        # TODO: 2. Initialize game state based on parameters

    def _reward_func(self, state, action_dict):
        # TODO: 1. Check to see if a player if frozen, if they are, ignore the action.
        # TODO: 2. If players are not frozen, if they use any of the four move actions,
        # move player to appropriate place, keeping track of walls + other player.
        # TODO 3. If a player moved, check if they collected an apple and return
        # the appropriate reward.

        # TODO: remove. Rock paper scissors below.
        agent_a, agent_b = action_dict.keys()[0], action_dict.keys()[1]
        action_a, action_b = action_dict[agent_a], action_dict[agent_b]

        # Win conditions.
        a_win = (action_a == "rock" and action_b == "scissors") or \
                (action_a == "paper" and action_b == "rock") or \
                (action_a == "scissors" and action_b == "paper")

        if action_a == action_b:
            reward_dict[agent_a], reward_dict[agent_b] = 0, 0
        elif a_win:
            reward_dict[agent_a], reward_dict[agent_b] = 1, -1
        else:
            reward_dict[agent_a], reward_dict[agent_b] = -1, 1

        return reward_dict



    def _transition_func(self, state, action):
        # TODO: 1. Repeat computations above & update player location if they moved.
        # TODO 2. If player rotates, update their direction in the state.
        # TODO 3. If a player shines a beam, see if it hits their opponent and
        # store the hit in the state.
        # TODO: 4. Generate apples based on parameters.
        # TODO: 5. Return the current state.

        # TODO: remove. Rock paper scissors below
        return state

    def __str__(self):
        return "gathering_game"
