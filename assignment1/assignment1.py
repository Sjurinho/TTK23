from matplotlib import pyplot as plt
from gridWorld import gridWorld
from copy import deepcopy
import numpy as np

def show_value_function(mdp, V):
    fig = mdp.render(show_state = False, show_reward = False)            
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        fig.axes[0].annotate("{0:.3f}".format(V[k]), (s[1] - 0.1, s[0] + 0.1), size = 40/mdp.board_mask.shape[0])
    plt.show(block=False)
    
def show_policy(mdp, PI):
    fig = mdp.render(show_state = False, show_reward = False)
    action_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        if mdp.terminal[s] == 0:
            fig.axes[0].annotate(action_map[PI[k]], (s[1] - 0.1, s[0] + 0.1), size = 100/mdp.board_mask.shape[0])
    plt.show(block=False)

####################  Problem 1: Value Iteration #################### 
def sum_of_states(mdp, V, state, action):
    return np.sum([mdp.transition_probability(state, action, s)*(mdp.reward(s) + gamma*V[s]) for s in mdp.states()])
def value_iteration(mdp, gamma, theta = 1e-3):
    # Make a valuefunction, initialized to 0
    V = np.zeros((len(mdp.states())))
    its = 0
    """
    YOUR CODE HERE:
    Problem 1a) Implement Value Iteration
    
    Input arguments:
        - mdp     Is the markov decision process, it has some usefull functions given below
        - gamma   Is the discount rate
        - theta   Is a small threshold for determining accuracy of estimation
    
    Some usefull functions of the grid world mdp:
        - mdp.states() returns a list of all states [0, 1, 2, ...]
        - mdp.actions(state) returns list of actions ["U", "D", "L", "R"] if state non-terminal, [] if terminal
        - mdp.transition_probability(s, a, s_next) returns the probability p(s_next | s, a)
        - mdp.reward(state) returns the reward of the state R(s)
    """
    max_deltV = 0
    V_old = np.zeros((len(mdp.states())))
    #raise Exception("Not implemented")
    while True:
        delta = 0
        V_old = deepcopy(V)
        for s in mdp.states():
            v = V[s]
            if len(mdp.actions(s))==0:
                V[s] = mdp.reward(s)
            else: 
                V[s] = np.max([np.sum([mdp.transition_probability(s, a, s_next)*(mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states()]) for a in mdp.actions(s)])
            delta = max(delta, abs(v - V[s]))
        if max_deltV < np.linalg.norm(V-V_old, np.inf):
            max_deltV = np.linalg.norm(V-V_old, np.inf)
        if delta < theta:
            break
        its += 1
    return V, max_deltV, its

def policy(mdp, V):
    # Initialize the policy list of crrect length
    PI = np.random.choice(env.actions(), len(mdp.states()))
    
    """
    YOUR CODE HERE:
    Problem 1b) Implement Policy function 
    
    Input arguments:
        - mdp Is the markov decision problem
        - V   Is the optimal falue function, found with value iteration
    """
    #raise Exception("Not implemented")
    for s in mdp.states():
        if len(mdp.actions(s))==0:
            PI[s] = 0
        else:
            PI[s] = mdp.actions(s)[np.argmax([np.sum([mdp.transition_probability(s, a, s_next)*(mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states()]) for a in mdp.actions(s)])]
    return PI

####################  Problem 2: Policy Iteration #################### 
def policy_evaluation(mdp, gamma, PI, V, theta = 1e-3):   
    """
    YOUR CODE HERE:
    Problem 2a) Implement Policy Evaluation
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor
        - PI    Is current policy
        - V     Is preveous value function guess
        - theta Is small threshold for determining accuracy of estimation
        
    Some useful tips:
        - If you decide to do exact policy evaluation, np.linalg.solve(A, b) can be used
          optionally scipy has a sparse linear solver that can be used
        - If you decide to do exact policy evaluation, note that the b vector simplifies
          since the reward R(s', s, a) is only dependant on the current state s, giving the 
          simplified reward R(s) 
    """
    n = len(mdp.states())
    A = np.zeros((n,n))
    b = np.zeros(n)
    for s in mdp.states():
        A[s] = gamma*np.array([mdp.transition_probability(s, PI[s], s_next) for s_next in mdp.states()])
        if len(mdp.actions(s))==0:
            b[s] = -mdp.reward(s)
        else:
            b[s] = -np.sum([mdp.transition_probability(s, PI[s], s_next)*mdp.reward(s) for s_next in mdp.states()])
    A[np.diag_indices(n)] -= 1
    V = np.linalg.solve(A, b)
    return V

def iterative_policy_evaluation(mdp, gamma, PI, V, theta = 1e-3):   
    """
    YOUR CODE HERE:
    Problem 2a) Implement Policy Evaluation
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor
        - PI    Is current policy
        - V     Is preveous value function guess
        - theta Is small threshold for determining accuracy of estimation
        
    Some useful tips:
        - If you decide to do exact policy evaluation, np.linalg.solve(A, b) can be used
          optionally scipy has a sparse linear solver that can be used
        - If you decide to do exact policy evaluation, note that the b vector simplifies
          since the reward R(s', s, a) is only dependant on the current state s, giving the 
          simplified reward R(s) 
    """
    V = np.zeros((len(mdp.states())))
    while True:
        delta = 0
        for i in range(len(mdp.states())):
            s = mdp.states()[i]
            v = V[i]
            if len(mdp.actions(s)) == 0:
                V[i] = mdp.reward(s)
            else:
                V[i] = sum(mdp.transition_probability(s, PI[s], s_next) * (mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states())
            delta = max(delta, abs(v - V[i]))
        if delta < theta:
            break

    #print(V)
    return V

def policy_iteration(mdp, gamma):
    # Make a valuefunction, initialized to 0
    V = np.zeros((len(mdp.states())))
    
    # Create an arbitrary policy PI
    PI = np.random.choice(env.actions(), len(mdp.states()))
    its = 0
    """
    YOUR CODE HERE:
    Problem 2b) Implement Policy Iteration
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor

    Some useful tips:
        - Use the the policy_evaluation function from the preveous subproblem
    """
    #raise Exception("Not implemented")
    max_deltV = 0
    V_OLD = np.zeros((len(mdp.states())))
    while True:
        PI_old = deepcopy(PI)
        V_OLD = deepcopy(V)
        V = policy_evaluation(mdp, gamma, PI, V)
        for i in range(len(mdp.states())):
            s = mdp.states()[i]
            if len(mdp.actions(s)) == 0:
                PI[i] = 0
            else:
                best_policy_idx = np.argmax([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states()) for a in mdp.actions(s)])
                PI[i] = mdp.actions(s)[best_policy_idx]
        if np.array_equal(PI, PI_old):
            break
        if max_deltV < np.linalg.norm(V-V_OLD, np.inf):
            max_deltV = np.linalg.norm(V-V_OLD, np.inf)
        its += 1
    return PI, V, max_deltV, its

if __name__ == "__main__":
    """
    Change the parameters below to change the behaveour, and map of the gridworld.
    gamma is the discount rate, while filename is the path to gridworld map. Note that
    this code has been written for python 3.x, and requiers the numpy and matplotlib
    packages

    Available maps are:
        - gridworlds/tiny.json
        - gridworlds/large.json
    """
    gamma   = 1
    filname = "gridworlds/tiny.json"


    # Import the environment from file
    env = gridWorld(filname)

    # Render image
    fig = env.render(show_state = False)
    plt.show()
    
    # Run Value Iteration and render value function and policy
    V, max_deltv, its = value_iteration(mdp = env, gamma = gamma)
    print(f'Value iteration:\nmax_deltV: {max_deltv}, its: {its}')
    show_value_function(env, V)
    
    PI = policy(env, V)
    show_policy(env, PI)
    
    # Run Policy Iteration and render value function and policy
    PI, V, max_deltv, its = policy_iteration(mdp = env, gamma = gamma)
    print(f'Policy iteration:\nmax_deltV: {max_deltv}, its: {its}')

    show_value_function(env, V)
    show_policy(env, PI)
    plt.show()
