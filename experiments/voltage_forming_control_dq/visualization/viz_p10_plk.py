import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from openmodelica_microgrid_gym.util import dq0_to_abc

from experiments.voltage_forming_control_dq.util.config import cfg
study_name = 'P10_Safe_DDPG_R_load_clip_state_clip_action' #cfg['STUDY_NAME']
"""
# SL ohne Delay
trial = ['13']
episode_list = ['0', '40']
terminated_list = ['0', '0']


# No SL ohne Delay
trial = ['19']
episode_list = ['0', '592']
terminated_list = ['1', '1']


# SL mit Delay
trial = ['20']
episode_list = ['0', '375']
terminated_list = ['1', '1']

"""
trial = ['9']
episode_list = ['0', '40', '160']
terminated_list = ['0', '0', '0']
study_name = 'P10_Safe_DDPG_R_load_delay' #cfg['STUDY_NAME']

interval_list_x = [0, 1]
ts = 1e-4

for i in range(len(trial)):

    #test_data = pd.read_pickle('data/experiment_data/' + study_name +'/' + trial[i] +'/' + trial[i] + '_'+study_name+'_' + trial[i]
    #                           +'_0.pkl.bz2')

    #train_data = pd.read_pickle('data/' + trial[i] + '_P10_SEC_R_load_traing_rewards_Trial_number_' + trial[i]

    train_data = pd.read_pickle('data/experiment_data/' + study_name +'/' + trial[i] +'/' + trial[i] + '_'+study_name+
                                '_training_rewards_trial_number'
                               +'_'+ trial[i]+'.pkl.bz2')


    """
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
    """


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

    plt.plot(train_data['Num_steps_per_episode'])
    plt.ylabel('num_Steps')
    plt.grid()
    plt.title(str(trial))
    plt.show()


    if episode_list is not None:
        for episode, terminated in zip(episode_list, terminated_list):
            episode_data = pd.read_pickle(
                'data/experiment_data/' + study_name + '/' + trial[i] + '/' + trial[i] + '_' + study_name +
                '_training_episode_number_'+ episode
                + '_terminated'+ terminated +'.pkl.bz2')

            fig, axs = plt.subplots(4, 1, figsize=(9, 7))

            t = np.arange(0, round((len(episode_data['i_a_training'].to_list())) * ts, 4), ts).tolist()

            v_ref = dq0_to_abc(np.array([325, 0, 0]), episode_data['Phase'].to_list())


            axs[1].plot(t, episode_data['i_a_training'].to_list(), 'b')#, label='$\mathrm{DDPG}_\mathrm{}$')
            axs[1].plot(t, episode_data['i_b_training'].to_list(), 'r')
            axs[1].plot(t, episode_data['i_c_training'].to_list(), 'g')
            axs[1].grid()
            #axs[0, 0].legend()
            #axs[0].set_title(['Episode' + str(episode)])
            #plt.xlim(interval_list_x)
            axs[1].set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{V}$')
            axs[3].set_xlabel(r'$t\,/\,\mathrm{s}$')
            #plt.show()

            axs[0].plot(t, episode_data['v_a_training'].to_list(), 'b')
            axs[0].plot(t, episode_data['v_b_training'].to_list(), 'r')
            axs[0].plot(t, episode_data['v_c_training'].to_list(), 'g')
            axs[0].plot(t, v_ref[0], ':', color='gray')
            axs[0].grid()
            #plt.legend()
            axs[0].set_title(['Episode' + str(episode)])
            #plt.xlim(interval_list_x)
            axs[0].set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')

            axs[2].plot(t, episode_data['v_a_training'].to_list(), 'b')
            axs[2].plot(t, episode_data['v_b_training'].to_list(), 'r')
            axs[2].plot(t, episode_data['v_c_training'].to_list(), 'g')
            axs[2].grid()
            axs[2].set_ylabel('$u_{\mathrm{abc}}\,/\,\mathrm{V}$')

            axs[3].plot(t, episode_data['Rewards_raw'].to_list(), 'b', label='$\mathrm{r}_\mathrm{env, unscaled}$')
            axs[3].plot(t, episode_data['Rewards_sum'].to_list(), 'r', label='$\mathrm{r}_\mathrm{punish, scaled}$')
            axs[3].grid()
            plt.legend(loc = "upper right")
            axs[3].set_ylabel('$u_{\mathrm{abc}}\,/\,\mathrm{V}$')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[3].text(0.25, 0.95, "$\mathrm{r}_\mathrm{min}(-0.75) \cdot (1-\gamma) = -0.04$", transform=axs[3].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            plt.show()

            asd = 1


