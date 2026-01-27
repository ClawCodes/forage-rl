import numpy as np
import mouse_maze
import model_based_rl
import q_learning_simple

filename = "mbrl_learning_trajectories.npy"
transitions = np.load(filename, allow_pickle=True)
print("Evaluating transitions from MBRL simulation")
maze = mouse_maze.SimpleMaze()
mbrl_algo = model_based_rl.MBRL(maze, num_episodes=200, gamma = 0.9)
mb_log_likelihood = mbrl_algo.simulate_model_based_rl(transitions)
print("Model-based RL log likelihood of transitions:", mb_log_likelihood)
qlearning_algo = q_learning_simple.QLearningTime(maze, num_episodes=200, alpha=0.1)
ql_log_likelihood = qlearning_algo.simulate_q_learning(transitions)
print("Q-learning log likelihood of transitions:", ql_log_likelihood)
if mb_log_likelihood > ql_log_likelihood:
    print("Model-based RL explains the data better")
else:
    print("Q-learning explains the data better")



filename = "q_learning_trajectories.npy"
transitions = np.load(filename, allow_pickle=True)
print("Evaluating transitions from Q-learning simulation")
maze = mouse_maze.SimpleMaze()
mbrl_algo = model_based_rl.MBRL(maze, num_episodes=200, gamma = 0.9)
mb_log_likelihood = mbrl_algo.simulate_model_based_rl(transitions)
print("Model-based RL log likelihood of transitions:", mb_log_likelihood)
qlearning_algo = q_learning_simple.QLearningTime(maze, num_episodes=200, alpha=0.1)
ql_log_likelihood = qlearning_algo.simulate_q_learning(transitions)
print("Q-learning log likelihood of transitions:", ql_log_likelihood)
if mb_log_likelihood > ql_log_likelihood:
    print("Model-based RL explains the data better")
else:
    print("Q-learning explains the data better")


