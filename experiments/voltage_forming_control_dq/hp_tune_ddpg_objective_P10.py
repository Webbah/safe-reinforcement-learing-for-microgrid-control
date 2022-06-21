import os
import time

import sqlalchemy
from optuna.samplers import TPESampler

from util.lr_scheduler import linear_schedule

os.environ['PGOPTIONS'] = '-c statement_timeout=1000'

import optuna
import platform
import argparse
import sshtunnel
import numpy as np
from experiments.voltage_forming_control_dq.util.config import cfg

from train_agent import train_ddpg

# np.random.seed(0)

PC2_LOCAL_PORT2PSQL = 11999
SERVER_LOCAL_PORT2PSQL = 6432
DB_NAME = 'optuna'
PC2_LOCAL_PORT2MYSQL = 11998
SERVER_LOCAL_PORT2MYSQL = 3306
STUDY_NAME = cfg['STUDY_NAME']  # 'DDPG_MRE_sqlite_PC2'

node = platform.uname().node


def ddpg_objective(trial):
    number_learning_steps = 500000  # trial.suggest_int("number_learning_steps", 100000, 1000000)
    actor_hidden_size = trial.suggest_int("actor_hidden_size", 10, 100)  # Using LeakyReLU
    actor_number_layers = trial.suggest_int("actor_number_layers", 1, 4)
    alpha_relu_actor = trial.suggest_loguniform("alpha_relu_actor", 0.001, 0.5)
    alpha_relu_critic = trial.suggest_loguniform("alpha_relu_critic", 0.001, 0.5)
    antiwindup_weight = trial.suggest_float("antiwindup_weight", 1e-4, 1)
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    bias_scale = trial.suggest_loguniform("bias_scale", 5e-5, 0.2)
    buffer_size = trial.suggest_int("buffer_size", int(20e4), number_learning_steps)  # 128
    critic_hidden_size = trial.suggest_int("critic_hidden_size", 10, 300)
    critic_number_layers = trial.suggest_int("critic_number_layers", 1, 4)
    error_exponent = 0.5  # 0.5  # trial.suggest_loguniform("error_exponent", 0.001, 4)
    final_lr = trial.suggest_float("final_lr", 0.00001, 1)
    gamma = trial.suggest_float("gamma", 0.6, 0.99999)
    integrator_weight = trial.suggest_float("integrator_weight", 1e-4, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1e-2)
    lr_decay_start = trial.suggest_float("lr_decay_start", 0.00001, 1)
    lr_decay_duration = trial.suggest_float("lr_decay_duration", 0.00001, 1)
    n_trail = str(trial.number)
    noise_steps_annealing = trial.suggest_int("noise_steps_annealing", int(0.1 * number_learning_steps),
                                              number_learning_steps)
    noise_theta = trial.suggest_loguniform("noise_theta", 1, 100)  # 25  # stiffness of OU
    noise_var = trial.suggest_loguniform("noise_var", 0.001, 1)  # 2
    noise_var_min = 0.0013  # trial.suggest_loguniform("noise_var_min", 0.0000001, 2)
    number_past_vals = trial.suggest_int("number_past_vals", 0, 20)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])  # , "LBFGS"])
    penalty_I_weight = trial.suggest_float("penalty_I_weight", 100e-6, 2)
    penalty_P_weight = trial.suggest_float("penalty_P_weight", 100e-6, 2)

    penalty_I_decay_start = trial.suggest_float("penalty_I_decay_start", 0.00001, 1)
    penalty_P_decay_start = trial.suggest_float("penalty_P_decay_start", 0.00001, 1)

    seed = 0

    t_start_penalty_I = int(penalty_I_decay_start * number_learning_steps)
    t_start_penalty_P = int(penalty_P_decay_start * number_learning_steps)
    t_start = int(lr_decay_start * number_learning_steps)
    t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                           number_learning_steps))
    tau = trial.suggest_loguniform("tau", 0.0001, 0.3)  # 2
    train_freq_type = "step"  # trial.suggest_categorical("train_freq_type", ["episode", "step"])
    training_episode_length = trial.suggest_int("training_episode_length", 1000, 4000)  # 128
    train_freq = trial.suggest_int("train_freq", 1, 5000)
    use_gamma_in_rew = 1
    weight_scale = trial.suggest_loguniform("weight_scale", 5e-5, 0.2)

    learning_rate = linear_schedule(initial_value=learning_rate, final_value=learning_rate * final_lr,
                                    t_start=t_start,
                                    t_end=t_end,
                                    total_timesteps=number_learning_steps)


    loss = train_ddpg(True, learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale,
                               alpha_relu_actor,
                               batch_size,
                               actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                               alpha_relu_critic,
                               noise_var, noise_theta, error_exponent,
                               training_episode_length, buffer_size,  # learning_starts,
                               tau, number_learning_steps, integrator_weight,
                               integrator_weight * antiwindup_weight, penalty_I_weight, penalty_P_weight,
                               train_freq_type, train_freq, t_start_penalty_I, t_start_penalty_P, optimizer,
                               number_past_vals, seed, n_trail)

    return loss


def get_storage(url, storage_kws):
    successfull = False
    retry_counter = 0

    while not successfull:
        try:
            storage = optuna.storages.RDBStorage(
                url=url, **storage_kws)
            successfull = True
        except (sqlalchemy.exc.OperationalError, sqlalchemy.exc.DatabaseError) as e:
            wait_time = np.random.randint(60, 300)
            retry_counter += 1
            if retry_counter > 10:
                print('Stopped after 10 connection attempts!')
                raise e
            print(f'Could not connect, retry in {wait_time} s')
            time.sleep(wait_time)

    return storage


def optuna_optimize_mysql_lea35(objective, sampler=None, study_name='dummy'):
    parser = argparse.ArgumentParser(description='Train DDPG Single Inverter V-ctrl')
    parser.add_argument('-n', '--n_trials', default=1, required=False,
                        help='number of trials to execute', type=int)
    args = parser.parse_args()
    n_trials = args.n_trials or 10

    print(n_trials)
    print('Local optimization is run - logs to MYSQL but measurement data is logged to MongoDB on Cyberdyne!')
    print('Take care, trail numbers can double if local opt. is run on 2 machines and are stored in '
          'the same MongoDB Collection!!!')
    print('Measurment data is stored to cfg[meas_data_folder] as json, from there it is grept via reporter to '
          'safely store it to ssh port for cyberdyne connection to mongodb')

    if node in ('lea-picard', 'lea-barclay'):
        creds_path = 'C:\\Users\\webbah\\Documents\\creds\\optuna_mysql.txt'
    else:
        # read db credentials
        creds_path = f'{os.getenv("HOME")}/creds/optuna_mysql'

    with open(creds_path, 'r') as f:
        optuna_creds = ':'.join([s.strip(' \n') for s in f.readlines()])

    if node in ('LEA-WORK35', 'fe1'):
        if node == 'fe1':
            port = PC2_LOCAL_PORT2MYSQL
        else:
            port = SERVER_LOCAL_PORT2MYSQL

        storage = get_storage(f'mysql://{optuna_creds}@localhost:{port}/{DB_NAME}')

        study = optuna.create_study(
            storage=storage,
            # storage=f'postgresql://{optuna_creds}@localhost:{port}/{DB_NAME}',
            sampler=sampler, study_name=study_name,
            load_if_exists=True,
            direction='maximize')
        study.optimize(objective, n_trials=n_trials)
    else:
        if node in cfg['lea_vpn_nodes']:
            # we are in LEA VPN
            server_name = 'lea35'
            tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                               SERVER_LOCAL_PORT2MYSQL)}
        else:
            # assume we are on a PC2 compute node
            server_name = 'fe.pc2.uni-paderborn.de'
            tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                               PC2_LOCAL_PORT2MYSQL),
                       'ssh_username': 'webbah'}
        with sshtunnel.open_tunnel(server_name, **tun_cfg) as tun:

            study = optuna.create_study(
                storage=f"mysql+pymysql://{optuna_creds}@127.0.0.1:{tun.local_bind_port}/{DB_NAME}",
                sampler=sampler, study_name=study_name,
                load_if_exists=True,
                direction='maximize')
            study.optimize(objective, n_trials=n_trials)


def optuna_optimize_mysql(objective, sampler=None, study_name='dummy'):
    parser = argparse.ArgumentParser(description='Train DDPG Single Inverter V-ctrl')
    parser.add_argument('-n', '--n_trials', default=1, required=False,
                        help='number of trials to execute', type=int)
    args = parser.parse_args()
    n_trials = args.n_trials or 10

    print(n_trials)
    print('Local optimization is run - logs to MYSQL but measurement data is logged to MongoDB on Cyberdyne!')
    print('Take care, trail numbers can double if local opt. is run on 2 machines and are stored in '
          'the same MongoDB Collection!!!')
    print('Measurment data is stored to cfg[meas_data_folder] as json, from there it is grept via reporter to '
          'safely store it to ssh port for cyberdyne connection to mongodb')

    if node in ('lea-picard', 'lea-barclay'):
        creds_path = 'C:\\Users\\webbah\\Documents\\creds\\optuna_mysql.txt'
    else:
        # read db credentials
        creds_path = f'{os.getenv("HOME")}/creds/optuna_mysql'

    with open(creds_path, 'r') as f:
        optuna_creds = ':'.join([s.strip(' \n') for s in f.readlines()])

    study = optuna.create_study(study_name=study_name,
                                direction='maximize',
                                storage=f"mysql://{optuna_creds}@localhost/{DB_NAME}",
                                load_if_exists=True,
                                sampler=sampler
                                )
    study.optimize(objective, n_trials=n_trials)


def optuna_optimize_sqlite(objective, sampler=None, study_name='dummy'):
    parser = argparse.ArgumentParser(description='Train DDPG Single Inverter V-ctrl')
    parser.add_argument('-n', '--n_trials', default=50, required=False,
                        help='number of trials to execute', type=int)
    args = parser.parse_args()
    n_trials = args.n_trials or 100

    print(n_trials)
    print('Local optimization is run but measurement data is logged to MongoDB on Cyberdyne!')
    print('Take care, trail numbers can double if local opt. is run on 2 machines and are stored in '
          'the same MongoDB Collection!!!')
    print('Measurment data is stored to cfg[meas_data_folder] as json, from there it is grept via reporter to '
          'safely store it to ssh port for cyberdyne connection to mongodb')

    if node in cfg['lea_vpn_nodes']:
        optuna_path = './optuna/'
    else:
        # assume we are on not of pc2 -> store to project folder
        optuna_path = '/scratch/hpc-prf-reinfl/weber/OMG/optuna/'

    os.makedirs(optuna_path, exist_ok=True)

    study = optuna.create_study(study_name=study_name,
                                direction='maximize',
                                storage=f'sqlite:///{optuna_path}optuna.sqlite',
                                load_if_exists=True,
                                sampler=sampler
                                )
    study.optimize(objective, n_trials=n_trials)



if __name__ == "__main__":
    TPE_sampler = TPESampler(n_startup_trials=1000)  # , constant_liar=True)

    #optuna_optimize_mysql_lea35(ddpg_objective, study_name=STUDY_NAME, sampler=TPE_sampler)
    optuna_optimize_sqlite(ddpg_objective, study_name=STUDY_NAME, sampler=TPE_sampler)
