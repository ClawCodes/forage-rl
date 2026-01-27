#value iteration with full observability
import numpy as np 


#TODO: make it connect to the MDP, for now I'm going to hard code everything

#Transition probs: #s0 (0.15), s1 (0.35), s2 (0.5) <-> s3 (0.15), s4 (0.35), s5 (0.5)

# self.decays = [0.5, 2.0, 0.1, 0.1, 2.0, 3.0]
# probs = np.exp(-self.decay * self.counter)

#S = states (6) + time spent in state (50)
num_patch_states = 6
max_time_spent = 10
gamma = 0.9
decays = [0.5, 2.0, 0.1, 0.1, 2.0, 3.0]
upper_patch = [0,1,2]
lower_patch = [3,4,5]

#value iteration
V = np.zeros((num_patch_states, max_time_spent))
delta = np.inf
while delta > 0.01:
    delta = 0
    for s in range(num_patch_states):
        for t in range(max_time_spent):
            #compute value of taking each action

            #Stay reward
            #reward is expected reward of taking the action
            r_sa = np.exp(-decays[s] * t)
            #calculate the probability of ending up in next state at time t+1
            #staying so no change in state but just +1 time spent
            if t + 1 < max_time_spent:
                val_stay = r_sa + gamma * V[s, t+1]
            else:
                val_stay = r_sa  # No future value if at max time

            #Leave reward
            #reward is always 0
            r_sa = 0
            val_leave = r_sa
            #add the discounted, transition weighted values for next states based on transition fn
            #Transition probs: #s0 (0.15), s1 (0.35), s2 (0.5) <-> s3 (0.15), s4 (0.35), s5 (0.5)
            if s in upper_patch:
                val_leave += gamma * (0.15 * V[3,0] + 0.35 * V[4,0] + 0.5 * V[5,0])
            else: #lower patch
                val_leave += gamma * (0.15 * V[0,0] + 0.35 * V[1,0] + 0.5 * V[2,0])
            
            V_new = max(val_stay, val_leave)
            delta = max(delta, np.abs(V_new - V[s, t]))
            V[s, t] = V_new

# Print the value function
print("Value Function:")
# Print the value function for each state and time step
for s in range(num_patch_states):
    print(f"State {s}:")
    for t in range(max_time_spent):
        print(f"  Time {t}: {V[s, t]:.2f}")
# Print the optimal policy
policy = np.zeros((num_patch_states, max_time_spent), dtype=int)
for s in range(num_patch_states):
    for t in range(max_time_spent):
        # Compute value for both actions
        r_stay = np.exp(-decays[s] * t)
        if t + 1 < max_time_spent:
            val_stay = r_stay + gamma * V[s, t+1]
        else:
            val_stay = r_stay

        val_leave = 0
        if s in upper_patch:
            val_leave += gamma * (0.15 * V[3,0] + 0.35 * V[4,0] + 0.5 * V[5,0])
        else:
            val_leave += gamma * (0.15 * V[0,0] + 0.35 * V[1,0] + 0.5 * V[2,0])

        # Choose the action with the higher value
        if val_stay >= val_leave:
            policy[s, t] = 0  # Stay
        else:
            policy[s, t] = 1  # Leave
# Print the policy
print("\nOptimal Policy:")
# Print the policy for each state and time step
for s in range(num_patch_states):
    print(f"State {s}:")
    for t in range(max_time_spent):
        action = "Stay" if policy[s, t] == 0 else "Leave"
        print(f"  Time {t}: {action}")

#print the expected value of starting in state 0 at time 0
print("\nExpected Value of starting in state 0 at time 0:")
print(V[0,0])