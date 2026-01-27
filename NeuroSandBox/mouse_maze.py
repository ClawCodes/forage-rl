import numpy as np

class foraging_reward():
    def __init__(self, decay):
        #exponential decay
        self.counter = 0
        self.decay = decay
    
    def reset(self):
        self.counter = 0
    
    def sample_reward(self):
        #compute probability of getting a reward
        prob = np.exp(-self.decay * self.counter)
        rand = np.random.rand()

        #incremenent counter to decrease food over time
        self.counter += 1

        #return stochastic reward
        if rand < prob:
            return 1.0
        else:
            return 0.0
        

class Maze:
    def __init__(self):
        #let's do 3 possibilities for reward dists for the top (upper patch) and 3 for the bottom (down) patch
        self.state = 0 #start at 0, later could randomize
        #action space is 0:stay, and 1:leave
        self.decays = [0.5, 2.0, 0.1, 0.1, 2.0, 3.0] #s2 best in upper patch, s3 best in lower patch
        # self.decays = [0, 0, 0, 100, 100, 100] #s2 best in upper patch, s3 best in lower patch
        #decays correspond to states 0,1,2,3,4,5
        #set up reward distributions
        self.rewards = [foraging_reward(d) for d in self.decays]
        self.horizon = 100 
        self.time = 0
        self.num_states = 6 #6 states
        self.num_actions = 2 #2 actions stay or leave

    def reset(self):
        #TODO: randomize start state 
        self.state = 0
        for r in self.rewards:
            r.reset()
        self.time = 0
        #return obs
        return self.state

    def step(self, action):
        #return reward based on the underlying latent state and the timestep since the agent went into a patch
        #calc transition
        new_state = self.get_transition(action)
        #calc reward
        reward = self.get_reward(new_state)
        #update state
        self.state = new_state

        #update time
        self.time += 1
        done = False
        if self.time >= self.horizon:
            done = True
        #return obs, reward, done
        return new_state, reward, done

    def get_reward(self, new_state):
        #provide reward based on the state and how long the mouse has been feeding at that state
        cur_state = self.state
        if cur_state == new_state:
            return self.rewards[cur_state].sample_reward()
        else:
            #mouse just left patch so reset all patches and return 0.0 reward since mouse traveling maze
            for r in self.rewards:
                r.reset()

            return 0.0

    def get_transition(self, action):
        #actions are 0:stay, 1:leave
        cur_state = self.state
        if action == 0: #stay
            return cur_state
        else:
            #perform a stochastic transitions
            #TODO: make it so it's not hard coded
            #s0 (0.15), s1 (0.35), s2 (0.5) <-> s3 (0.15), s4 (0.35), s5 (0.5)
            if cur_state in [0,1,2]:
                #stochastic dynamics
                rand = np.random.rand()
                if rand < 0.15:
                    return 3
                elif rand < 0.5:
                    return 4
                else:
                    return 5
                
            else:
                #cur_state in [3,4,5]
                #stochastic dynamics
                rand = np.random.rand()
                if rand < 0.15:
                    return 0
                elif rand < 0.5:
                    return 1
                else:
                    return 2
    

class MazePOMDP(Maze):
    def __init__(self):
        super().__init__()
        self.num_states = 2  #observation space is 0:upper patch, 1:lower patch
        self.num_actions = 2 #action space is 0:stay, and 1:leave

    def step(self, action):
        true_state, reward, done = super().step(action)  
        if true_state in [0,1,2]:
            obs = 0 #upper patch
        else:
            obs = 1 #lower patch
        return obs, reward, done
        


class SimpleMaze:
    def __init__(self):
        #just one reward on top and one on bottom
        self.state = 0 #start at 0, later could randomize
        #action space is 0:stay, and 1:leave
        self.decays = [0.2, 3.0] #upper patch is better than lower
       #set up reward distributions
        self.rewards = [foraging_reward(d) for d in self.decays]
        self.horizon = 100 
        self.time = 0
        self.num_states = 2 #2 states
        self.num_actions = 2 #2 actions stay or leave

    def reset(self):
        #TODO: randomize start state 
        self.state = 0
        for r in self.rewards:
            r.reset()
        self.time = 0
        #return obs
        return self.state

    def step(self, action):
        #return reward based on the underlying latent state and the timestep since the agent went into a patch
        #calc transition
        new_state = self.get_transition(action)
        #calc reward
        reward = self.get_reward(new_state)
        #update state
        self.state = new_state

        #update time
        self.time += 1
        done = False
        if self.time >= self.horizon:
            done = True
        #return obs, reward, done
        return new_state, reward, done

    def get_reward(self, new_state):
        #provide reward based on the state and how long the mouse has been feeding at that state
        cur_state = self.state
        if cur_state == new_state:
            return self.rewards[cur_state].sample_reward()
        else:
            #mouse just left patch so reset all patches and return 0.0 reward since mouse traveling maze
            for r in self.rewards:
                r.reset()

            return 0.0

    def get_transition(self, action):
        #actions are 0:stay, 1:leave
        cur_state = self.state
        if action == 0: #stay
            return cur_state
        else:
            #transition deterministically to other patch
            if cur_state in [0]:
                return 1
                
            else:
                return 0
            


        
if __name__ == "__main__":
    #debugging code
    print("starting experiment")
    maze_env = SimpleMaze()
    # maze_pomdp = MazePOMDP()
    print("current state", maze_env.state)
    actions = [0,0,0,0,0,1,0,0,0,0]
    for step in range(len(actions)):
        print("---- step ", step)
        action = actions[step]
        if action == 0:
            print("taking action stay")
        else:
            print("taking action leave")
        obs, r, done = maze_env.step(action)
        print("next state", obs, "reward", r, "done", done)

    # print("starting POMDP experiment")
    # maze_pomdp = MazePOMDP()
    # print("current state", maze_env.state)
    # for step in range(10):
    #     print("---- step ", step)
    #     if np.random.rand() < 0.5:
    #         action = 0 #stay
    #         print("taking action stay")
    #     else:
    #         action = 1
    #         print("taking action leave")
    #     obs, r = maze_pomdp.step(action)
    #     print("next state", obs, "reward", r)

    # for j, r in enumerate(maze_env.rewards):
    #     r.reset()
    #     print('reward samples from distribution', j)
    #     for i in range(20):
           
    #         print(r.sample_reward())

