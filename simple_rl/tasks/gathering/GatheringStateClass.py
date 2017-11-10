''' GatheringStateClass.py: Contains the GatheringState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class GatheringState(State):

    def __init__(self, x1, y1, x2, y2, apples, hits):
        # Locations of player 1 and player 2
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

        # Locations of the apples. TODO: List of coordinates?
        self.apples = apples

        # Number of successive hits for each player
        # (1 - potential to be frozen, 2 - frozen)
        self.hits = hits

        # TODO: add beams

        # TODO: add each players direction (pose)


    def __hash__(self):
        # TODO: Will we need this?
        pass

    def __str__(self):
        # TODO: Print string reprensentation of the State.
        pass

    def __eq__(self, other):
        # TODO
        pass

    def to_rgb(self):
        pass

        # TODO: Do we want this function? Converts the state to an RGB reprensentation
        # that can be used by the DQN learner. (Maybe create a superclass of State)
        # that has this method?
