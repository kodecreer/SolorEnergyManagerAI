import gymnasium as gym
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import env #Leave this in so it registers the environment


if __name__ == '__main__':


    # env = gym.make('CartPole-v1', num_envs=2)
    envs = gym.make('SolarEnv-v0')

   
    # Reset the environment to its initial state and receive the initial observation
    observation = envs.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 10

    test_size = 2000
    random.seed(40) #For consistency #Seeds 40, 40, 41
    test_inds = random.sample(range(0, 365*48), test_size)
    testx = []
    test_tmp = []
    testy = []
    for episode in tqdm(range(num_episodes)):
        observation, _ = envs.reset()
        done = [False]
        #When not done. This is an array of 
        #dones
        step = 0
        total = 0
        while step < 365 * 48:
            actions = 2#random.randint(1, 2)
            if step in test_inds:
                #Perform calculations without gradients
            
                next_observation, reward, done,truncated,  _ = envs.step(actions)
                test_tmp.append(reward )
          

                # Update the current observation with the next observation
                observation = next_observation
            else:
                envs.step(actions) #On hold only
            step += 1
        
        testx.append(episode)
        testy.append(sum(test_tmp) / len(test_tmp))
       

    plt.title("Sell only")
    plt.scatter(testx, testy)
    print(testx, testy)
    plt.savefig('r.pdf', bbox_inches='tight')  
    # Close the environment when done
    # print(sum(agent.memory.rewards[-1]))
    envs.close()