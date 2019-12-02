from sample_players import DataPlayer

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
        #bestmove = self.alpha_beta_search(state)
        #print(bestmove)
        #random = random.choice(state.actions())
        #print(random)
        
        print("CALL")
        best_move = None
        for depth in range(1, self.depth_limit+1):
            best_move = self.alpha_beta_search(state, depth)
            self.queue.put(best_move)



    def alpha_beta_search(self, gameState, depth):
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
            value = self.min_value(gameState.result(action), depth - 1, alpha, beta)
            alpha = max(alpha, value)
            return value

        return max(gameState.actions(), key=lambda x: alphaBeta(x, alpha))

    # TODO: modify the function signature to accept an alpha and beta parameter


    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def min_value(self, state, depth, alpha, beta):
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