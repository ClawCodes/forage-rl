import mouse_maze
import numpy as np
import matplotlib.pyplot as plt


#Model-based RL using value iteration to compute optimal policy under assumption of known dynamics but will learn rewards
class MBRL():
    def __init__(self,maze, num_episodes, gamma = 0.9, horizon = 10, beta=1.0, filename=None):
        self.maze = maze
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.num_planning_steps = horizon
        self.beta = beta # Inverse temperature for softmax action selection
        self.q_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))  # Q-values for each state-action pair
        self.r_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))  # Reward estimates for each state-action pair
        self.count = np.zeros((maze.num_states, maze.horizon, maze.num_actions))  # Count of visits for each state-action pair
        self.filename = filename

    def q_value_iteration(self):
        #perform q-value iteration using learned reward and known transition function
        for step in range(self.num_planning_steps):
            # print("planning step", step)
            for s in range(self.maze.num_states):
                for t in range(self.maze.horizon):
                    for a in range(self.maze.num_actions):
                        r_sa = self.r_table[s, t, a]
                        #get next state based on transition function
                        if a == 0: #stay
                            next_state = s
                            next_time = min(t + 1, self.maze.horizon - 1)
                        else: #leave
                            #stochastic transitions
                            if s == 0:
                                next_state = 1
                            else:
                                next_state = 0
                            next_time = 0  # reset time when leaving
                        self.q_table[s,t,a] = r_sa + self.gamma * np.max(self.q_table[next_state, next_time])


    def simulate_model_based_rl(self, transitions):
        ''' Simulate model-based RL using the loaded transitions and return the loglikelihood of these transitions under model-based RL'''
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

            # Update reward table with running average
            self.count[state, time_spent, action] += 1
            self.r_table[state, time_spent, action] += (reward - self.r_table[state, time_spent, action]) / self.count[state, time_spent, action]
            # After each transition, perform planning using value iteration
            self.q_value_iteration()

        return log_likelihoods

    def train(self):
        transitions = []
        for episode in range(self.num_episodes):
            print("episode", episode)
            state = self.maze.reset()
            time_spent = 0
            done = False
            while not done:
                #pick action based on Boltzmann exploration of current Q-values using softmax
                action_probs = np.exp(self.q_table[state, time_spent] * self.beta)  # Temperature parameter
                action_probs /= np.sum(action_probs)  # Normalize probabilities
                action = np.random.choice(range(self.maze.num_actions), p=action_probs)  # Sample action
                next_state, reward, done = self.maze.step(action)

                #write out transition to npy file for analysis later
                transition = (state, time_spent, action, reward, next_state)
                transitions.append(transition)

                # Update reward table with running average
                self.count[state, time_spent, action] += 1
                self.r_table[state, time_spent, action] += (reward - self.r_table[state, time_spent, action]) / self.count[state, time_spent, action]

                if state == next_state:
                    time_spent += 1
                else:
                    time_spent = 0
                state = next_state

                # After each transition, perform planning using value iteration
                self.q_value_iteration()

        #save transitions
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
        mbrl_learning = MBRL(maze, num_episodes=6, gamma = 0.9, filename = f"mbrl_learning_trajectories_{i}.npy")

        mbrl_learning.train()
    #TODO: debug this. not sure if resluts make sense
    # q_learning_time = QLearningTime(maze, num_episodes=4000, alpha = 0.1)
    # q_learning_time.train()

    #unpack and print the saved transitions
    # transitions = np.load(mbrl_learning.filename, allow_pickle=True)
    # for i, transition in enumerate(transitions):
    #     print(f"Transition {i}: {transition}")  