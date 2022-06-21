import json
import platform
from datetime import time
from os import makedirs
from time import sleep

import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openmodelica_microgrid_gym.util import abc_to_dq0
from stable_baselines3 import DDPG

from experiments.voltage_forming_control_dq.envs.env_wrapper import SecWrapperPastVals, BaseWrapper, SecWrapper
from experiments.voltage_forming_control_dq.envs.rewards import Reward
from experiments.voltage_forming_control_dq.envs.vctrl_single_inv import net
from experiments.voltage_forming_control_dq.util.config import cfg
from experiments.voltage_forming_control_dq.util.custom_network import custom_network

# np.random.seed(0)

folder_name = cfg['STUDY_NAME']
node = platform.uname().node

max_eps_steps = 600
used_model = 'model.zip'
def execute_ddpg(gamma, use_gamma_in_rew, alpha_relu_actor, actor_number_layers, error_exponent,
               training_episode_length,
              number_learning_steps, integrator_weight, antiwindup_weight,
               number_past_vals, n_trial):


    if node in cfg['lea_vpn_nodes']:
        save_folder = 'data/' + cfg['meas_data_folder']
        log_path = f'{folder_name}/{n_trial}/'
    else:
        # assume we are on a node of pc2 -
        save_folder = cfg['pc2_logpath'] + '/' + cfg['meas_data_folder']
        pc2_log_path = cfg['pc2_logpath']
        log_path = f'{pc2_log_path}/{folder_name}/{n_trial}/'

    makedirs(save_folder, exist_ok=True)

    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent, i_lim=net['inverter1'].i_lim,
                 i_nom=net['inverter1'].i_nom)

    #env_test = gym.make('experiments.voltage_forming_control_dq.envs:vctrl_single_inv_test-v1',
    env_test = gym.make('experiments.voltage_forming_control_dq.envs:vctrl_single_inv_train-v0',
                        reward_fun=rew.rew_fun_dq0,
                        abort_reward=-1,  # no needed if in rew no None is given back
                        obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                    'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                    'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2'],
                        max_episode_steps=max_eps_steps
                        )

    if cfg['env_wrapper'] == 'past':
        env_test = SecWrapperPastVals(env_test, number_of_features=9 + number_past_vals * 3,
                                      integrator_weight=integrator_weight,
                                      # recorder=mongo_recorder,
                                      antiwindup_weight=antiwindup_weight,
                                      gamma=1, penalty_I_weight=0,
                                      penalty_P_weight=0, number_past_vals=number_past_vals,
                                      training_episode_length=training_episode_length, config=cfg, safe=1)

    elif cfg['env_wrapper'] == 'no-I-term':
        env_test = BaseWrapper(env_test, number_of_features=6 + number_past_vals * 3,
                               training_episode_length=training_episode_length,
                               # recorder=mongo_recorder,
                               n_trail=n_trial, gamma=gamma,
                               number_learing_steps=number_learning_steps, number_past_vals=number_past_vals)

    else:
        env_test = SecWrapper(env_test, number_of_features=11, integrator_weight=integrator_weight,
                              # recorder=mongo_recorder,
                              antiwindup_weight=antiwindup_weight,
                              gamma=1, penalty_I_weight=0,
                              penalty_P_weight=0,
                              training_episode_length=training_episode_length, )

    # increase action space to generate model with 6 outputs if SEC
    if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
        env_test.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

    print('Before load')

    model = DDPG.load(log_path + f'{used_model}', env=env_test)

    print('After load')

    #model = custom_network(model, actor_number_layers, critic_number_layers,
    #                       weight_scale, bias_scale, alpha_relu_actor, alpha_relu_critic)

    count = 0
    for kk in range(actor_number_layers + 1):

        if kk < actor_number_layers:
            model.actor.mu._modules[str(count + 1)].negative_slope = alpha_relu_actor
            model.actor_target.mu._modules[str(count + 1)].negative_slope = alpha_relu_actor

        count = count + 2

    if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
        env_test.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))
    ####### Run Test Case#########
    return_sum = 0.0
    rew.gamma = 0
    # episodes will not abort, if limit is exceeded reward = -1
    rew.det_run = True
    rew.exponent = 0.5  # 1
    limit_exceeded_in_test = False

    # , use_past_vals=True, number_past_vals=30)
    # using gamma=1 and rew_weigth=3 we get the original reward from the env without penalties
    obs = env_test.reset()
    phase_list = []
    phase_list.append(env_test.env.net.components[0].phase)

    rew_list = []
    v_d = []
    v_q = []
    v_0 = []
    action_P0 = []
    action_P1 = []
    action_P2 = []
    action_I0 = []
    action_I1 = []
    action_I2 = []
    integrator_sum0 = []
    integrator_sum1 = []
    integrator_sum2 = []
    R_load = []

    for step in range(env_test.max_episode_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)
        action_P0.append(np.float64(action[0]))
        action_P1.append(np.float64(action[1]))
        action_P2.append(np.float64(action[2]))
        if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
            action_I0.append(np.float64(action[3]))
            action_I1.append(np.float64(action[4]))
            action_I2.append(np.float64(action[5]))
            integrator_sum0.append(np.float64(env_test.integrator_sum[0]))
            integrator_sum1.append(np.float64(env_test.integrator_sum[1]))
            integrator_sum2.append(np.float64(env_test.integrator_sum[2]))
        if cfg['loglevel'] in ['train', 'test']:
            phase_list.append(env_test.env.net.components[0].phase)

        if rewards == -1 and not limit_exceeded_in_test:  # and env_test.rew[-1]:
            # Set addidional penalty of -1 if limit is exceeded once in the test case
            limit_exceeded_in_test = True

        if limit_exceeded_in_test:
            # if limit was exceeded once, reward will be kept to -1 till the end of the episode,
            # nevertheless what the agent does
            rewards = -1

        env_test.render()
        return_sum += rewards
        rew_list.append(rewards)

        if step % 1000 == 0 and step != 0:
            env_test.close()
            obs = env_test.reset()

        if done:
            env_test.close()
            break

    env_test.close()

    v_a = env_test.history.df['lc.capacitor1.v']
    v_b = env_test.history.df['lc.capacitor2.v']
    v_c = env_test.history.df['lc.capacitor3.v']
    i_a = env_test.history.df['lc.inductor1.i']
    i_b = env_test.history.df['lc.inductor2.i']
    i_c = env_test.history.df['lc.inductor3.i']
    R_load = (env_test.history.df['r_load.resistor1.R'].tolist())
    phase = env_test.history.df['inverter1.phase.0']  # env_test.env.net.components[0].phase
    v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
    i_dq0 = abc_to_dq0(np.array([i_a, i_b, i_c]), phase)

    i_d = i_dq0[0].tolist()
    i_q = i_dq0[1].tolist()
    i_0 = i_dq0[2].tolist()
    v_d = (v_dq0[0].tolist())
    v_q = (v_dq0[1].tolist())
    v_0 = (v_dq0[2].tolist())

    t = np.arange(0, round((len(v_0)) * 1e-4, 4), 1e-4).tolist()


    plt.plot(t, v_d)
    plt.ylabel('v_d')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.plot(v_q)
    plt.ylabel('v_q')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.plot(v_0)
    plt.ylabel('v_0')
    plt.xlabel('t')
    plt.grid()
    plt.show()

    sleep(1)

    plt.plot(t, i_d)
    plt.ylabel('i_d')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.plot(i_q)
    plt.ylabel('i_q')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.plot(i_0)
    plt.ylabel('i_0')
    plt.xlabel('t')
    plt.grid()
    plt.show()

    ts = time.gmtime()
    compare_result = {"Name": "comparison_PI_DDPG",
                      "time": ts,
                      "ActionP0": action_P0,
                      "ActionP1": action_P1,
                      "ActionP2": action_P2,
                      "ActionI0": action_I0,
                      "ActionI1": action_I1,
                      "ActionI2": action_I2,
                      "integrator_sum0": integrator_sum0,
                      "integrator_sum1": integrator_sum1,
                      "integrator_sum2": integrator_sum2,
                      "DDPG_model_path": log_path,
                      "Return DDPG": (return_sum / env_test.max_episode_steps),
                      "Reward DDPG": rew_list,
                      "env_hist_DDPG": env_test.env.history.df,
                      "info": "execution of RL agent on 10 s test case-loading values",
                      "optimization node": 'Thinkpad',
                      }
    store_df = pd.DataFrame([compare_result])
    store_df.to_pickle(f'{folder_name}/' + used_model + f'_{max_eps_steps}steps')


    return return_sum / env_test.max_episode_steps


file_congfig = open(
    'experiments/voltage_forming_control_dq/PC2_DDPG_Vctrl_single_inv_22_newTestcase_Trial_number_11534_0.json', )
trial_config = json.load(file_congfig)
execute_ddpg(trial_config["gamma"], 1, trial_config["alpha_relu_actor"], trial_config["actor_number_layers"], 0.5,
             trial_config["training_episode_length"], 1000000, trial_config["integrator_weight"],
             trial_config["antiwindup_weight"], 5, 0)
