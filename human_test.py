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
    num_episodes = 1

    test_size = 100
    random.seed(40) #For consistency
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
        while step < 365 * 48:
            
            if step in test_inds:
                #Perform calculations without gradients
                actions = -1
                try:
                    while actions < 0 or actions > 1:
                        print('Price')
                        #Account for weekend
                        WATTAGE_RATE = envs.energy_prices[envs.energy_step]
                        print(WATTAGE_RATE)
                        print('year, month, day, hour, minute, irradiance, temperature, Solar Zenith Angle, Wind Speed, Relative Humidity, Vmp, Imp')
                        print(observation)
                        actions = int(input("Sell (0) or Hold (1)"))
                except:
                    actions = 1
                next_observation, reward, done,truncated,  _ = envs.step(actions)
                test_tmp.append(reward )
          

                # Update the current observation with the next observation
                observation = next_observation
            step += 1
        envs.step(0) #On hold only
        testx.append(episode)
        testy.append(sum(test_tmp) / len(test_tmp))
       

    plt.title("Human only with Hold only")
    plt.scatter(testx, testy)
    plt.savefig('human_metrics.pdf', bbox_inches='tight')  
    # Close the environment when done
    # print(sum(agent.memory.rewards[-1]))
    envs.close()