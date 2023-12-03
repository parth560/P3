# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            # Create a new set of values to hold the updated values for this iteration
            newValues = util.Counter()

            # For each state, we update the value using the Bellman equation
            for state in self.mdp.getStates():
                # Checking if the current state is not a terminal state
                if not self.mdp.isTerminal(state):
                    # Computing the maximum Q-value for the current state by considering all possible actions
                    newValues[state] = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))

            # Now Updating the values to the new values for the next iteration
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # The sum of the products of probabilities and the discounted future values.
        qValue = sum(prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action))
        # sum iterates over all possible next states and their probabilities
        return qValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Check if the current state is a terminal state and if so then no action can be taken.
        if self.mdp.isTerminal(state):
            return None

        # Obtaining the list of possible actions in the current state.
        possibleActions = self.mdp.getPossibleActions(state)
        # The action with the maximum Q-value using the computeQValueFromValues function.
        bestAction = max(possibleActions, key=lambda action: self.computeQValueFromValues(state, action))
        # Return the action with the highest Q-value in the current state.
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        # Determine the total number of states
        num_states = len(states)
        # Select the current state based on the iteration count through all states.
        for i in range(self.iterations):
            state = states[i % num_states]  
            # Check if the current state is not a terminal state.
            if not self.mdp.isTerminal(state):
            # Update the value for the current state using the Q-value of the best action.
                self.values[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pQueue = util.PriorityQueue()

        predecessors = {}

        #find all predecessors
        for state in self.mdp.getStates():
            # Check if the current state is not a terminal state
            if not self.mdp.isTerminal(state):
                 # Iterate over all possible actions in the current state
                for action in self.mdp.getPossibleActions(state):
                    # Iterate over all possible next states and their probabilities
                    for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):
                        # Check if the next state is already in the predecessors dictionary
                        if stateAndProb[0] in predecessors:
                            # If yes, add the current state as a predecessor
                            predecessors[stateAndProb[0]].add(state)
                        else:
                            # If not create a new set with the current state as the only predecessor
                            predecessors[stateAndProb[0]] = {state}

        # Iterate through all states to calculate the differences between current values
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Calculate the absolute difference between the current value and the maximum Q-value.
                diff = abs(self.values[state] - max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]))
            # Push the negative difference into the priority queue
                pQueue.update(state, -diff)

        # Iterate through a specified number of iterations for value iteration.
        for iteration in range(self.iterations):
            # Check if the priority queue is empty if so then break the loop.
            if pQueue.isEmpty():
                break
            # Pop a state with the largest Q-value change from the priority queue.
            state = pQueue.pop()
            if not self.mdp.isTerminal(state):
            # Update the value of the current state based on the maximum Q-value achievable from possible actions.
                self.values[state] = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])

            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                # Calculate the absolute difference between the current value and the maximum Q-value for the predecessor.
                    diff = abs(self.values[p] - max([self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)]))
                    # Update the queue if the difference exceeds a threshold
                    if diff > self.theta:
                        pQueue.update(p, -diff)