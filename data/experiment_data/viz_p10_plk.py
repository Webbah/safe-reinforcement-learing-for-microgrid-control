import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


trial = ['1']

interval_list_x = [0, 1]
ts = 1e-4

for i in range(len(trial)):

    test_data = pd.read_pickle(trial[i] + '_DDPG_single_train_bestStudy22_test_reward_trial_number_' + trial[i]
                               +'.pkl.bz2')

    train_data = pd.read_pickle(trial[i] + '_DDPG_single_train_bestStudy22_training_rewards_trial_number_' + trial[i]
                               +'.pkl.bz2')

    test_reward = test_data['Reward']
    t_reward = np.arange(0, round((len(test_reward)) * ts, 4), ts).tolist()

    plt.plot(t_reward, np.array(test_reward), 'b', label=f'      SEC-DDPG: '
                                            f'{round(sum(test_reward[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')


    plt.grid()
    plt.xlim(interval_list_x)
    plt.legend()
    plt.ylabel("Reward")
    plt.show()

    plt.plot(test_data['Reward'])
    plt.ylabel('Test Reward')
    plt.grid()
    plt.title(str(trial))
    plt.show()

    plt.plot(train_data['Mean_eps_env_reward_raw'])
    plt.ylabel('Train Reward raw (w/o penalty)')
    plt.grid()
    plt.title(str(trial))
    plt.show()

    plt.plot(train_data['Mean_eps_reward_sum'])
    plt.ylabel('Train Reward sum')
    plt.grid()
    plt.title(str(trial))
    plt.show()

    asd = 1

