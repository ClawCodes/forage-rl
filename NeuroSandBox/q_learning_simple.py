import mouse_maze
import numpy as np
import matplotlib.pyplot as plt


#Q-learning algorithm that incorporates time spent in state as part of the state representation
class QLearningTime:
    def __init__(self, maze, num_episodes, alpha=0.05, gamma=0.9, epsilon=0.1, beta=1.0, filename=None):
        self.maze = maze
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta # Inverse temperature for softmax action selection
        self.q_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))  # Q-values for each state-action pair
        self.filename = filename

    def choose_action(self, state, time_spent):
        #use boltzmann exploration to choose action
        action_probs = np.exp(self.q_table[state, time_spent] * self.beta)
        action_probs /= np.sum(action_probs)  # Normalize probabilities
        action = np.random.choice(range(self.maze.num_actions), p=action_probs)  # Sample action
        return action
        
    def update_q_value(self, state, time_spent, action, reward, next_state):
        if state == next_state:
            next_time_spent = min(time_spent + 1, self.maze.horizon - 1)
        else:
            next_time_spent = 0
        best_next_action = np.argmax(self.q_table[next_state, next_time_spent])
        td_target = reward + self.gamma * self.q_table[next_state, next_time_spent, best_next_action]
        td_error = td_target - self.q_table[state, time_spent, action]
        self.q_table[state][time_spent][action] += self.alpha * td_error

    def simulate_q_learning(self, transitions):
        ''' Simulate Q-learning using the loaded transitions and return the loglikelihood of these transitions under Q-learning updates'''
        log_likelihoods = []

        for i, transition in enumerate(transitions):
            # print("transition", i)
            state, time_spent, action, reward, next_state = transition
            #convert to int for indexing
            state = int(state)
            time_spent = int(time_spent)
            action = int(action)
            next_state = int(next_state)

            # Compute the log likelihood of the transition under the current policy assuming a Boltzmann policy
            action_probs = np.exp(self.q_table[state, time_spent] * self.beta)
            action_probs /= np.sum(action_probs)  # Normalize probabilities
            log_likelihoods.append(np.log(action_probs[action]))

            # Update Q table based on transition
            self.update_q_value(state, time_spent, action, reward, next_state)

        return log_likelihoods


    def train(self):
        transitions = []
        for episode in range(self.num_episodes):
            state = self.maze.reset()
            time_spent = 0
            done = False
            max_time_spent_during_episode = 0
            while not done:
                action = self.choose_action(state, time_spent)
                next_state, reward, done = self.maze.step(action)

                #write out transition to npy file for analysis later
                transition = (state, time_spent, action, reward, next_state)
                transitions.append(transition)
                # print("state", state, "time spent", time_spent, "action", action, "reward", reward, "next state", next_state)
                # Update Q-value
                self.update_q_value(state, time_spent, action, reward, next_state)
                if state == next_state:
                    time_spent += 1
                else:
                    time_spent = 0
                state = next_state
                max_time_spent_during_episode = max(max_time_spent_during_episode, time_spent)
            print("episode", episode, "max time spent during episode", max_time_spent_during_episode)
            # print out metric to see if Q-values are converging
            if episode % 100 == 0:
                avg_q_value = np.mean(self.q_table)
                print(f"Episode {episode}, Average Q-value: {avg_q_value:.4f}")


        #save transitions   
        # only save if filename is provided
        if self.filename is not None:   
            print("Saving transitions to", self.filename) 
            np.save(self.filename, transitions)

        print("Training completed.")
        # print the optimal policy for each state and for time spent up to 4 steps
        max_time_to_display = min(6, self.maze.horizon)
        policy = np.zeros((self.maze.num_states, self.maze.horizon), dtype=int) 
        for s in range(self.maze.num_states):
            for t in range(self.maze.horizon):
                policy[s, t] = np.argmax(self.q_table[s, t])
        print("Optimal Policy:")
        for s in range(self.maze.num_states):
            print(f"State {s}:")
            for t in range(max_time_to_display):
                action = "Stay" if policy[s, t] == 0 else "Leave"
                print(f"  Time {t}: {action}")
        # print("Final Q-table:")
        # print(self.q_table)
        # self.plot_q_values()
    def plot_q_values(self, max_time_to_display=6):
        max_time = min(max_time_to_display, self.maze.horizon)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        actions = ["Stay", "Leave"]
        # Find global min and max for consistent color scale
        vmin = np.min(self.q_table[:, :max_time, :])
        vmax = np.max(self.q_table[:, :max_time, :])
        for i, action in enumerate([0, 1]):
            im = axes[i].imshow(self.q_table[:, :max_time, action], vmin=vmin, vmax=vmax)
            axes[i].set_title(f"Q-values for Action {action} ({actions[i]})")
            axes[i].set_xlabel("Time Spent")
            axes[i].set_ylabel("States")
            axes[i].set_xticks(range(max_time))
            axes[i].set_xticklabels([f"t={j}" for j in range(max_time)])
            axes[i].set_yticks(range(self.maze.num_states))
            axes[i].set_yticklabels(["Upper Patch", "Lower Patch"])
            # Annotate each cell with its Q-value
            for y in range(self.maze.num_states):
                for x in range(max_time):
                    value = self.q_table[y, x, action]
                    axes[i].text(x, y, f"{value:.2f}", ha="center", va="center", color="w" if (vmax-vmin)>0 and (value-vmin)/(vmax-vmin)>0.5 else "black", fontsize=8)
            fig.colorbar(im, ax=axes[i])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    for i in range(100):
        maze = mouse_maze.SimpleMaze()
        q_learning = QLearningTime(maze, num_episodes=6, alpha=0.1, filename=f"q_learning_trajectories_{i}.npy")

        q_learning.train()
    #TODO: debug this. not sure if resluts make sense
    # q_learning_time = QLearningTime(maze, num_episodes=4000, alpha = 0.1)
    # q_learning_time.train()

    #unpack and print the saved transitions
    # transitions = np.load(q_learning.filename, allow_pickle=True)
    # for i, transition in enumerate(transitions):
    #     print(f"Transition {i}: {transition}")  