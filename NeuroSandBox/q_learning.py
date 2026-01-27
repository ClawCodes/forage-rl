from mouse_maze import MazePOMDP
import numpy as np
import matplotlib.pyplot as plt

#ignores time spent in state
class QLearning:
    def __init__(self, maze, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, min_epsilon=0.01, decay_rate=0.995):
        self.maze = maze
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_table = np.zeros((maze.num_states, 2))  # Q-values for each state-action pair
        self.qs = [[] for _ in range(maze.num_states*maze.num_actions)]  # Store Q-values for each state
        self.returns = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Explore: choose a random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: choose the action with the highest Q-value

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def train(self):
        for episode in range(self.num_episodes):
            state = self.maze.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.maze.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            self.qs[0].append(self.q_table[0][0]) #debugging to see if Q-values are converging
            self.qs[1].append(self.q_table[0][1]) #debugging to see if Q-values are converging
            self.qs[2].append(self.q_table[1][0]) #debugging to see if Q-values are converging
            self.qs[3].append(self.q_table[1][1]) #debugging to see if Q-values are converging
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.returns.append(total_reward)
        print("Training completed.")
        print("Final Q-table:")
        print(self.q_table)
        
        self.plot_q_values()
        self.plot_q_values_over_time()
        self.plot_returns()
        plt.show()

    def plot_returns(self):
        plt.figure()
        plt.plot(self.returns)
        plt.title("Returns over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Return")

    def plot_q_values_over_time(self):
        plt.figure()
        print("length", len(self.qs))
        plt.plot(self.qs[0], label="O1 Stay")
        plt.plot(self.qs[1], label="O1 Leave")
        plt.plot(self.qs[2], label="O2 Stay")
        plt.plot(self.qs[3], label="O2 Leave")
        plt.title("Q-values over time")
        plt.legend()
        # plt.show()

    def plot_q_values(self):
        plt.imshow(self.q_table)
        plt.colorbar()
        plt.title("Q-values")
        plt.xlabel("Actions")
        plt.xticks([0, 1], ["Stay", "Leave"])
        plt.ylabel("States")
        plt.yticks(range(self.maze.num_states), ["Upper Patch", "Lower Patch"])
        plt.tight_layout()
        # plt.show()

#Q-learning algorithm that incorporates time spent in state
#TODO: need to debug this, not sure if time spent is being handled correctly
class QLearningTime:
    def __init__(self, maze, num_episodes, alpha=0.05, gamma=0.9, epsilon=0.1):
        self.maze = maze
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))  # Q-values for each state-action pair

    def choose_action(self, state, time_spent):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Explore: choose a random action
        else:
            return np.argmax(self.q_table[state, time_spent])  # Exploit: choose the action with the highest Q-value

    def update_q_value(self, state, time_spent, action, reward, next_state):
        if state == next_state:
            next_time_spent = min(time_spent + 1, self.maze.horizon - 1)
        else:
            next_time_spent = 0
        best_next_action = np.argmax(self.q_table[next_state, next_time_spent])
        td_target = reward + self.gamma * self.q_table[next_state, next_time_spent, best_next_action]
        td_error = td_target - self.q_table[state, time_spent, action]
        self.q_table[state][time_spent][action] += self.alpha * td_error

    def train(self):
        for episode in range(self.num_episodes):
            state = self.maze.reset()
            time_spent = 0
            done = False
            max_time_spent_during_episode = 0
            while not done:
                action = self.choose_action(state, time_spent)
                next_state, reward, done = self.maze.step(action)
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

                

        print("Training completed.")
        # print the optimal policy for each state and for time spent up to 4 steps
        max_time_to_display = min(4, self.maze.horizon)
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
        self.plot_q_values()
    def plot_q_values(self, max_time_to_display=4):
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
    maze = MazePOMDP()
    q_learning = QLearning(maze, num_episodes=5000,alpha=0.0005,epsilon=1.0,min_epsilon=0.01, decay_rate=1.0)

    q_learning.train()
    #TODO: debug this. not sure if resluts make sense
    # q_learning_time = QLearningTime(maze, num_episodes=4000, alpha = 0.1)
    # q_learning_time.train()