
import numpy as np
import matplotlib.pyplot as plt

#import logprob data saved from model_inference_experiment.py and visualize results
#use logprobabilities to compute accuracy of model-based RL vs Q-learning in explaining data from each model
def plot_model_comparison(num_datasets):
    mb_accuracies = []
    ql_accuracies = []

    for i in range(num_datasets):
        mb_log_likelihoods = np.load(f"./logprobs/mbrl_true_log_likelihoods_{i}.npy")
        ql_log_likelihoods = np.load(f"./logprobs/ql_false_log_likelihoods_{i}.npy")

        if mb_log_likelihoods[-1] > ql_log_likelihoods[-1]:
            mb_accuracies.append(1)
        else:
            mb_accuracies.append(0)

        mb_log_likelihoods_ql = np.load(f"./logprobs/mbrl_false_log_likelihoods_{i}.npy")
        ql_log_likelihoods_ql = np.load(f"./logprobs/ql_true_log_likelihoods_{i}.npy")

        if ql_log_likelihoods_ql[-1] > mb_log_likelihoods_ql[-1]:
            ql_accuracies.append(1)
        else:
            ql_accuracies.append(0)

    #Plot accuracies
    print([np.mean(mb_accuracies), np.mean(ql_accuracies)])
    plt.bar(['Model-Based RL', 'Q-Learning'], [np.mean(mb_accuracies), np.mean(ql_accuracies)])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison Accuracy')
    plt.show()


#plot how accuracy changes as function of cumulative sum of loglikelihoods
def plot_cumulative_sum_accuracy(num_datasets):
    accuracies = []
    for j in range(num_datasets):
        # print(j)
        mb_cum_log_likelihoods = np.load(f"./logprobs/mbrl_true_log_likelihoods_{j}.npy")
        ql_cum_log_likelihoods = np.load(f"./logprobs/ql_false_log_likelihoods_{j}.npy")
        accuracy_mb = np.zeros(len(mb_cum_log_likelihoods))
        for i in range(len(mb_cum_log_likelihoods)):
            if np.isclose(mb_cum_log_likelihoods[i], ql_cum_log_likelihoods[i]):
                # print("tie detected")
                accuracy_mb[i] = 0.5    
            elif mb_cum_log_likelihoods[i] > ql_cum_log_likelihoods[i]:
                accuracy_mb[i] = 1
            else:
                accuracy_mb[i] = 0
        accuracies.append(accuracy_mb)

        mb_log_likelihoods_ql = np.load(f"./logprobs/mbrl_false_log_likelihoods_{j}.npy")
        ql_log_likelihoods_ql = np.load(f"./logprobs/ql_true_log_likelihoods_{j}.npy")
        accuracy_ql = np.zeros(len(mb_log_likelihoods_ql))
        for i in range(len(mb_log_likelihoods_ql)):
            if np.isclose(mb_log_likelihoods_ql[i], ql_log_likelihoods_ql[i]):
                # print("tie detected")
                accuracy_ql[i] = 0.5
            elif ql_log_likelihoods_ql[i] > mb_log_likelihoods_ql[i]:
                accuracy_ql[i] = 1
            else:
                accuracy_ql[i] = 0
        accuracies.append(accuracy_ql)

    #make a pretty plot of average accuracy across datasets with thicker lines
    #also make the font nice sized for everything
    avg_accuracy = np.mean(accuracies, axis=0)
    plt.plot(avg_accuracy, linewidth=3)
    plt.ylim(0.4, 1)
    plt.xlabel('Number of Observed Transitions', fontsize=16)
    plt.ylabel('Prediction Accuracy', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


if __name__ == "__main__":
    plot_model_comparison(num_datasets=100)
    plot_cumulative_sum_accuracy(num_datasets=100)
    plt.show()