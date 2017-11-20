''' GatheringDilemmaMDPClass.py: Contains an implementation of Gathering from
the Deep Mind paper Multi-agent Reinforcement Learning in Sequential Social
Dilemmas. '''

# Python imports.
import random
import numpy as np

# Other imports.
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from simple_rl.tasks.gathering.GatheringStateClass import GatheringState
from simple_rl.tasks.gathering.GatheringStateClass import GatheringAgent

# TODO: remove when State is changed to GatheringState
from simple_rl.mdp.StateClass import State



INITIAL_ORIENTATION = 'NORTH'
ROTATION_MATRICES = {
    'NORTH' : np.matrix([[1, 0],[0, 1]]),
    'SOUTH' : np.matrix([[-1, 0],[0, -1]]),
    'EAST' : np.matrix([[0, -1],[1, 0]]),
    'WEST' : np.matrix([[0, 1],[-1, 0]]),
}
MOVEMENT_VECTOR = {
    'step_forward' : np.matrix([[0], [-1]]),
    'step_backward' : np.matrix([[0], [1]]),
    'step_left' : np.matrix([[-1], [0]]),
    'step_right' : np.matrix([[1], [0]]),
}
ROTATE_LEFT = {
    'NORTH' : 'WEST',
    'SOUTH' : 'EAST',
    'EAST' : 'NORTH',
    'WEST' : 'SOUTH',
}
ROTATE_LEFT = {
    'NORTH' : 'EAST',
    'SOUTH' : 'WEST',
    'EAST' : 'SOUTH',
    'WEST' : 'NORTH',
}

class GatheringMDP(MarkovGameMDP):

    # Static constants.
    ACTIONS = ["step_forward", "step_backward", "step_left", "step_right", "rotate_left", "rotate_right", "use_beam", "stand_still"]

    def __init__(self, gamma, possible_apple_locations, N_apples, N_tagged, dim=(35, 13)):
        self.gamma, self.N_apples, self.N_tagged = gamma, N_apples, N_tagged
        self.x_dim, self.y_dim = dim[0], dim[1]

        agent1 = GatheringAgent(31, 6, False, INITIAL_ORIENTATION, 0, 0)
        agent2 = GatheringAgent(32, 5, False, INITIAL_ORIENTATION, 0, 0)

        idx = np.array(possible_apple_locations)
        print(idx)

        initial_apple_locations = np.zeros(shape=[self.x_dim, self.y_dim], dtype=np.int32)
        initial_apple_locations[idx[:, 0], idx[:, 1]] = 1
        print(initial_apple_locations)

        MarkovGameMDP.__init__(
            self,
            GatheringMDP.ACTIONS,
            self._transition_func,
            self._reward_func,
            init_state=GatheringState(agent1, agent2, initial_apple_locations),
        )


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
        agent_a, agent_b = action_dict.keys()[0], action_dict.keys()[1]
        action_a, action_b = action_dict[agent_a], action_dict[agent_b]

        # This function iterates the time step for everything
        # Iterates frozen_time_remaining, updates when apples will regenerate
        # also gives pointers to the new agent objects
        nextState, agent_a, agent_b = self.next_time_step(state, agent_a, agent_b)

        # TODO: Set action to None if an agent cannot move
        if agent_a.frozen_time_remaining > 0:
            action_a = None
        if agent_b.frozen_time_remaining > 0:
            action_b = None

        if action_a.startswith('step') and action_b.startswith('step'):
            if self._can_perform_move(agent_a, action_a) and self._can_perform_move(agent_b, action_b):
                a_x, a_y = self._get_next_location(agent_a, action_a)
                b_x, b_y = self._get_next_location(agent_b, action_b)
                if a_x == b_x and a_y == b_y:
                    if random.random() > 0.5:
                        agent_a.x, agent_a.y = a_x, a_y
                    else:
                        agent_b.x, agent_b.y = b_x, b_y
                    return newState

        for agent, act in [(agent_a, action_a), (agent_b, action_b)]:
            if act.startswith('step') and self._can_perform_move(agent, act):
                pos_x, pos_y = self._get_next_location(agent, act)
                agent.x, agent.y = pos_x, pos_y
            elif act == 'rotate_left':
                agent.orientation = ROTATE_LEFT[agent.orientation]
            elif act == 'rotate_right':
                agent.orientation = ROTATE_RIGHT[agent.orientation]
            elif act =='use_beam':
                agent.is_shining = ! agent.is_shining

        # TODO 3. If a player shines a beam, see if it hits their opponent and
        # store the hit in the state.
        if action_a == 'use_beam':
            pass
        if action_b == 'use_beam'
            pass

        # TODO: 4. Generate apples based on parameters.
        ## ^^ should this happen before people move? What if I move into a square
        ## where an apple was going to be regenerated? Does it not appear until I move?
        ## or do I go ahead and get the points?

        # TODO: 5. Return the current state.

        # TODO: remove. Rock paper scissors below
        return state

    def _can_perform_move(self, agent, action):
        final_pos_x, final_pos_y = self._get_next_location(agent, action)
        return final_pos_x > 0 and final_pos_x < x_dim - 1 and final_pos_y > 0 and final_pos_y < y_dim - 1

    def _get_next_location(self, agent, action):
        movement = np.multiply(ROTATION_MATRICES[agent.orientation], MOVEMENT_VECTOR[action])
        return agent.x + movement[0], agent.y + movement[1]

    def _update_apples(self, state):
        pass

    def _update_state(self, agent, is_shining=None, orientation=None, hits=None):


    def __str__(self):
        return "gathering_game"
