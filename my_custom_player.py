from sample_players import DataPlayer
from threading import Lock
from isolation import DebugState, Isolation
from multiprocessing import Process, Value, Lock, Manager
import math
from enum import IntEnum

class MyContext():
    def __init__(self):
        self.node_counter = 0
        self.level_counter = 0


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    depth_limit = 9

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
       
        if self.context is None:
            self.context = MyContext()
        print(" MY GO: "+str(state.ply_count) +
              " Levels "+str(self.context.level_counter) +
              " Nodes: "+str(self.context.node_counter))

        best_move = None

        if state.ply_count < 2:
            best_move = random.choice(state.actions())
            self.queue.put(best_move)
        else:
            #minmax
            #best_move = self.minimax(state, depth=self.depth_limit)
            #self.queue.put(best_move)

            #alpha-beta
            for depth in range(1, self.depth_limit+1):
                best_move = self.alpha_beta_search(state, depth)
                self.context.level_counter = depth
                self.queue.put(best_move)



    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")

        def alphaBeta(action, alpha):
            value = self.min_value(state.result(action), depth - 1, alpha, beta)
            alpha = max(alpha, value)
            return value

        return max(state.actions(), key=lambda x: alphaBeta(x, alpha))

    # def addState(self, state):
    #     self.context.states[state.locs] = state

    def addNode(self, state):
        self.context.node_counter += 1

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        return len(own_liberties) - len(opp_liberties)
        # return self.customHeuristic(state)
    
    center = (5, 4)
    max_distance = 7
    def customHeuristic(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        #The maximum distance a liberty can have from center point is 6
        #Start by giving each liberty 7, then reduce distance
        total_own = len(own_liberties) * self.max_distance
        total_opp = len(opp_liberties) * self.max_distance

        for lib in own_liberties:
            cell = DebugState.ind2xy(lib)
            total_own -= int(math.hypot(self.center[0] - cell[0], self.center[1] - cell[1]))

        for lib in opp_liberties:
            cell = DebugState.ind2xy(lib)
            total_opp -= int(math.hypot(self.center[0] - cell[0], self.center[1] - cell[1]))

        #print("Total own "+str(total_own)+" Total opp "+str(total_opp))
        return total_own - total_opp


    def min_value(self, state, depth, alpha, beta):
        self.addNode(state)
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        value = float("inf")
        for action in state.actions():
            value = min(value, self.max_value(state.result(action), depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value(self, state, depth, alpha, beta):
        self.addNode(state)
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.min_value(state.result(action), depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)        
        return value



    def minimax(self, state, depth):

        def min_value(state, depth):
            self.addNode(state)
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("inf")
            #print("MIN Action"+" Node " + str(state.ply_count)+" Depth:"+str(depth))
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            self.addNode(state)
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("-inf")
            #print("MAX Action"+" Node "+str(state.ply_count)+" Depth:"+str(depth))
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def sit_on_opposition(self, state):
        myLocation = state.locs[state.player()]
        apple = Isolation(board=state.board,
                          ply_count=state.ply_count + 1, locs=state.locs)
        actions = apple.actions()
        print(actions)
        if myLocation is not None:
            return actions[0]
        else:
            return DebugState.ind2xy(int(actions[0]))

