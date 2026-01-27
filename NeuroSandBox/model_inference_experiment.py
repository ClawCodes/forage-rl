import numpy as np
import mouse_maze
import model_based_rl
import q_learning_simple

maze = mouse_maze.SimpleMaze()

#iterate through files saved from model-based RL and Q-learning simulations and evaluate loglikelihood of transitions under both models
model_based_predictions = [] #1 correct, 0 incorrect
cum_sum_accuracy = []
for i in range(100):
    filename = f"mbrl_learning_trajectories_{i}.npy"
    transitions = np.load(filename, allow_pickle=True)
    print()
    print("Evaluating transitions from MBRL simulation from file:", filename)
    mbrl_algo = model_based_rl.MBRL(maze, num_episodes=200, gamma = 0.9)
    mb_log_likelihoods = mbrl_algo.simulate_model_based_rl(transitions)
    print("Model-based RL log likelihood of transitions:", np.sum(mb_log_likelihoods))
    qlearning_algo = q_learning_simple.QLearningTime(maze, num_episodes=200, alpha=0.1)
    ql_log_likelihoods = qlearning_algo.simulate_q_learning(transitions)
    print("Q-learning log likelihood of transitions:", np.sum(ql_log_likelihoods))

    #for each transition in the dataset, compare loglikelihoods under mb and q-learning to predict
    #first compute cumulative sum of loglikelihoods
    #then see for this dataset if mb has higher cumulative sum than q-learning
    mb_cumsum = np.cumsum(mb_log_likelihoods)
    ql_cumsum = np.cumsum(ql_log_likelihoods)

    #Save these values for later analysis
    np.save(f"./logprobs/mbrl_true_log_likelihoods_{i}.npy", mb_cumsum)
    np.save(f"./logprobs/ql_false_log_likelihoods_{i}.npy", ql_cumsum)

    # accuracy_mb = mb_cumsum > ql_cumsum
    # cum_sum_accuracy.append(accuracy_mb)


    # if np.sum(mb_log_likelihoods) > np.sum(ql_log_likelihoods):
    #     print("Model-based RL explains the data better")
    #     model_based_predictions.append(1)
    # else:
    #     print("Q-learning explains the data better")
    #     model_based_predictions.append(0)
    # np.save(f"model_based_accuracy.npy", np.array(model_based_predictions))


qlearning_based_predictions = [] #1 correct, 0 incorrect
for i in range(100):
    filename = f"q_learning_trajectories_{i}.npy"
    transitions = np.load(filename, allow_pickle=True)
    print()
    print("Evaluating transitions from Q-learning simulation from file:", filename)
    maze = mouse_maze.SimpleMaze()
    mbrl_algo = model_based_rl.MBRL(maze, num_episodes=200, gamma = 0.9)
    mb_log_likelihoods = mbrl_algo.simulate_model_based_rl(transitions)
    print("Model-based RL log likelihood of transitions:", np.sum(mb_log_likelihoods))
    qlearning_algo = q_learning_simple.QLearningTime(maze, num_episodes=200, alpha=0.1)
    ql_log_likelihoods = qlearning_algo.simulate_q_learning(transitions)
    print("Q-learning log likelihood of transitions:", np.sum(ql_log_likelihoods))

    #for each transition in the dataset, compare loglikelihoods under mb and q-learning to predict
    #first compute cumulative sum of loglikelihoods
    #then see for this dataset if q-learning has higher cumulative sum of loglikelihoods
    mb_cumsum = np.cumsum(mb_log_likelihoods)
    ql_cumsum = np.cumsum(ql_log_likelihoods)

    #Save these values for later analysis
    np.save(f"./logprobs/mbrl_false_log_likelihoods_{i}.npy", mb_cumsum)
    np.save(f"./logprobs/ql_true_log_likelihoods_{i}.npy", ql_cumsum)

    # accuracy_ql = ql_cumsum > mb_cumsum
    # cum_sum_accuracy.append(accuracy_ql)

    # if np.sum(mb_log_likelihoods) > np.sum(ql_log_likelihoods):
    #     print("Model-based RL explains the data better")
    #     qlearning_based_predictions.append(0)
    # else:
    #     print("Q-learning explains the data better")
    #     qlearning_based_predictions.append(1)
    # np.save(f"qlearning_based_accuracy.npy", np.array(qlearning_based_predictions))

    # np.save(f"cumulative_sum_accuracy.npy", np.array(cum_sum_accuracy))


