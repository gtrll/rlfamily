import os
import time
import functools
import copy
import git
import tensorflow as tf
import numpy as np
from rl import policies as Pol
from rl import envs as Env
from rl import oracles as Or
from rl import algorithms as Alg
from rl.adv_estimators import AdvantageEstimator
from rl.experimenter.generate_rollouts import generate_rollout
from rl.online_learners import online_optimizer as OO
from rl.tools import normalizers as Nor
from rl.tools import supervised_learners as Sup
from rl.tools.online_learners.scheduler import PowerScheduler
from rl.tools.online_learners import base_algorithms as bAlg
from rl.tools.utils import logz


def configure_log(configs, unique_log_dir=False):
    """ Configure output directory for logging. """

    # parse configs to get log_dir
    c = configs['general']
    top_log_dir = c['top_log_dir']
    log_dir = c['exp_name']
    seed = c['seed']

    # create dirs
    os.makedirs(top_log_dir, exist_ok=True)
    if unique_log_dir:
        log_dir += '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(top_log_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, '{}'.format(seed))
    os.makedirs(log_dir, exist_ok=True)

    # Log commit number.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    configs['git_commit_sha'] = sha

    # save configs
    logz.configure_output_dir(log_dir)
    logz.save_params(configs)


def general_setup(c):
    envid, seed = c['envid'], c['seed'],
    env = Env.create_env(envid, seed)
    # fix randomness
    tf.set_random_seed(seed)  # graph-level seed
    np.random.seed(seed)
    return env, envid, seed


def create_policy(env, seed, c, name='learner_policy'):
    pol_cls = getattr(Pol, c['policy_cls'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    build_nor = Nor.create_build_nor_from_str(c['nor_cls'], c['nor_kwargs'])
    policy = pol_cls(ob_dim, ac_dim, name=name, seed=seed,
                     build_nor=build_nor, **c['pol_kwargs'])
    return policy


def create_advantage_estimator(policy, adv_configs, name='value_function_approximator'):
    adv_configs = copy.deepcopy(adv_configs)
    # create the value function SupervisedLearner
    c = adv_configs['vfn_params']
    build_nor = Nor.create_build_nor_from_str(c['nor_cls'], c['nor_kwargs'])
    vfn_cls = getattr(Sup, c['fun_class'])
    vfn = vfn_cls(policy.ob_dim, 1, name=name, build_nor=build_nor, **c['fun_kwargs'])
    # create the adv object
    adv_configs.pop('vfn_params')
    adv_configs['vfn'] = vfn
    ae = AdvantageEstimator(policy, **adv_configs)
    return ae


def create_oracle(policy, ae, c):
    assert c['loss_type'] == 'rl'
    nor = Nor.NormalizerStd(None, **c['nor_kwargs'])
    oracle = Or.tfPolicyGradient(policy, ae, nor, **c['or_kwargs'])
    return oracle


def create_model_oracle(oracle, env, envid, seed, c):
    mor_cls = getattr(Or, c['mor_cls'])
    if mor_cls is Or.SimulationOracle:
        et = c['env_type']
        seed = seed + 1
        if et == 'true':
            sim_env = Env.create_env(envid, seed)
        elif et == 'mlp':
            sim_env = Env.create_sim_env(env, seed, dyn_configs=c['dynamics'])
        else:
            assert isinstance(et, float) and et >= 0.0 and et < 1.0
            sim_env = Env.create_sim_env(env, seed, inaccuracy=et)
        gen_ro = functools.partial(generate_rollout, env=sim_env, **c['rollout_kwargs'])
        model_oracle = mor_cls(oracle, sim_env, gen_ro)
    elif (mor_cls is Or.LazyOracle) or (mor_cls is Or.AdversarialOracle):
        model_oracle = mor_cls(oracle, **c['lazyor_kwargs'])
    elif mor_cls is Or.AggregatedOracle:
        model_oracle = mor_cls(Or.LazyOracle(oracle, **c['lazyor_kwargs']), **c['aggor_kwargs'])
    elif mor_cls is Or.DummyOracle:
        model_oracle = mor_cls(oracle)

    else:
        raise ValueError('Unknown model oracle type.')

    return model_oracle


def create_algorithm(policy, oracle, c, **kwargs):
    alg_cls = getattr(Alg, c['alg_cls'])
    if alg_cls is Alg.PredictiveRL:
        # create piccolo
        scheduler = PowerScheduler(**c['learning_rate'])
        x0 = policy.variable
        # create base_alg
        if c['base_alg'] == 'adam':
            base_alg = bAlg.Adam(x0, scheduler, beta1=c['adam_beta1'])
        elif c['base_alg'] == 'adagrad':
            base_alg = bAlg.Adagrad(x0, scheduler, rate=c['adagrad_rate'])
        elif c['base_alg'] == 'natgrad':
            base_alg = bAlg.AdaptiveSecondOrderUpdate(x0, scheduler)
        elif c['base_alg'] == 'trpo':
            base_alg = bAlg.TrustRegionSecondOrderUpdate(x0, scheduler)

        else:
            raise ValueError('Unknown base_alg')
        # create piccolo
        if c['n_model_steps'] is None:  # the very basic piccolo
            if c['base_alg'] in ['trpo', 'natgrad']:  # use Fisher information matrix
                pcl = OO.PiccoloFisherReg(policy, base_alg, p=c['learning_rate']['p'], use_shift=c['use_shift'],
                                          damping=c['reg_damping'])
            else:
                pcl = OO.Piccolo(policy, base_alg, p=c['learning_rate']['p'])
        else:  # multiple-step version
            assert c['n_model_steps'] >= 0
            if c['base_alg'] in ['trpo', 'natgrad']:  # use Fisher information matrix
                method = c['model_step_learning_rate']
                pcl = OO.PiccoloOptBasicFisherReg(policy, base_alg,
                                                  p=c['learning_rate']['p'], n_steps=c['n_model_steps'],
                                                  method=method, use_shift=c['use_shift'],
                                                  damping=c['reg_damping'])
            else:
                pcl = OO.PiccoloOptBasic(policy, base_alg,
                                         p=c['learning_rate']['p'], n_steps=c['n_model_steps'],
                                         method=c['model_step_learning_rate'])

        # create algorithm
        alg = alg_cls(pcl, oracle, policy, model_oracle=kwargs['model_oracle'], **c['alg_kwargs'])
    else:
        raise ValueError('Unknown algorithm type.')
    return alg
