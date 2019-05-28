import copy
from rl.tools.utils.misc_utils import dict_update

configs = {
    'general': {
        'top_log_dir': 'log',
        'envid': 'DartCartPole-v1',
        'seed': 0,
        'exp_name': 'cp',
    },
    'experimenter': {
        'run_alg_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
        },
        'rollout_kwargs': {
            'min_n_samples': 4000,
            'max_n_rollouts': None,
            'max_rollout_len': None,  # the max length of rollouts in training
        },
    },
    'algorithm': {
        'alg_cls': 'PredictiveRL',
        'n_model_steps': None,
        'opt_method': 'BFGS',
        'model_step_learning_rate': None,  # if it's None, it reuses the current base_alg; otherwise it's should be a dict
        'alg_kwargs': {
            'update_rule': 'model-free',  # 'piccolo', 'model-free', 'model-based', 'dyna'
            'update_in_pred': False,
            'take_first_pred': False,
            'warm_start': True,  # for VI
            'shift_adv': False,  # for VI
            'stop_std_grad': False,
            'ignore_samples': False,  # for reproducing classic model-based algorithms
        },
        'base_alg': 'adam',  # 'natgrad', 'adam', 'adagrad', 'trpo'
        'adam_beta1': 0.9,
        'adagrad_rate': 0.5,
        'reg_damping': 0.1,
        'use_shift': False,
        'learning_rate': {
            'p': 0,
            'eta': 0.01,
            'c': 0.01,
            'limit': None,
        },
    },
    'oracle': {
        'loss_type': 'rl',
        'or_kwargs': {
            'avg_type': 'sum',  # 'avg' or 'sum'
            'correlated': True,  # False, # update adv nor before computing adv
            'use_log_loss': False,
            'normalize_weighting': True,
        },
        'nor_kwargs': {  # simple baseline substraction
            'rate': .0,
            'momentum': 0,  # 0 for instant, None for moving average
            'clip_thre': None,
            'unscale': True,
        },
    },
    'model_oracle': {
        'mor_cls': 'LazyOracle',  # 'SimulationOracle', 'LazyOracle', 'AggregatedOracle', 'DummyOracle' 'AdversarialOracle',
        # 'true',  # 'true': true env; [0, 1): custom env with inaccurate env; 'mlp': custom env with mlp dyn
        'env_type': 0.1,
        'dyn_learning_with_piccolo_weight': False,  # not useful yet
        'dyn_learning_weights_type': 'one',  # 'one' or 'T-t', way to weight the importance of samples in a rollout
        'lazyor_kwargs': {
            'beta': 0.,  # mixing coefficient of using a lazy oracle
        },
        'ensemble_size': 1,  # for weighted oracle
        'weighting_mode': 'average',  # 'average', # 'recent', # for weighted oracle
        'aggor_kwargs': {
            'max_n_rollouts': None,
            'max_n_samples': None,
            'max_n_iterations': 3,
        },
        'learn_dyn': False,
        'rollout_kwargs': {
            'min_n_samples': 4000,
            'max_n_rollouts': None,
            'max_rollout_len': None,
        },
    },
    'policy': {
        'nor_cls': 'tfNormalizerMax',
        'nor_kwargs': {
            'rate': 0.0,
            'momentum': None,
            'clip_thre': 5.0,
        },
        'policy_cls': 'tfGaussianMLPPolicy',
        'pol_kwargs': {
            'size': 32,  # 64,
            'n_layers': 1,  # 2
            'init_logstd': -1.0,
            'max_to_keep': 10,
        },
    },
    'advantage_estimator': {
        'gamma': 1.0,
        'delta': 0.99,  # discount to make MDP well-behave, or variance reduction parameter
        'lambd': 0.98,   # 0, 0.98, GAE parameter, in the estimation of on-policy value function
        'default_v': 0.0,  # value ofn the absorbing state
        # 'monte-carlo' (lambda = 1), 'td' (lambda = 0), 'same' (as GAE lambd),
        # or TD(lambda) for learning vf.
        'v_target': 0.98,
        'onestep_weighting': False,  # whether to use one-step importance weight (only for value function learning)
        'multistep_weighting': False,  # whether to use multi-step importance weight
        'data_aggregation': False,
        'max_n_rollouts': None,  # for data aggregation
        'n_updates': 1,
        'vfn_params': {
            'nor_cls': 'tfNormalizerMax',
            'nor_kwargs': {
                'rate': 0.0,
                'momentum': None,
                'clip_thre': 5.0,
            },
            'fun_class': 'tfMLPSupervisedLearner',
            'fun_kwargs': {
                'learning_rate': 1e-3,
                'batch_size': 128,  
                'n_batches': 2048,
                'batch_size_for_prediction': 2048,
                'size': 64,
                'n_layers': 2,
            },
        },
    },
}

# Hopper.
t = {
    'general': {
        'exp_name': 'hopper',
        'envid': 'DartHopper-v1',
    },
    'experimenter': {
        'run_alg_kwargs': {'n_itrs': 200},
        'rollout_kwargs': {'min_n_samples': 16000},
    },
    'model_oracle': {
        'rollout_kwargs': {'min_n_samples': 16000},
    },
}
configs_hopper = copy.deepcopy(configs)
dict_update(configs_hopper, t)


# Snake.
t = {
    'general': {
        'exp_name': 'snake',
        'envid': 'DartSnake7Link-v1',
    },
    'experimenter': {
        'run_alg_kwargs': {'n_itrs': 200},
        'rollout_kwargs': {'min_n_samples': 16000},
    },
    'model_oracle': {
        'rollout_kwargs': {'min_n_samples': 16000},
    },
}
configs_snake = copy.deepcopy(configs)
dict_update(configs_snake, t)


# Walker3d.
t = {
    'general': {
        'exp_name': 'walker3d',
        'envid': 'DartWalker3d-v1',
    },
    'experimenter': {
        'run_alg_kwargs': {'n_itrs': 1000},
        'rollout_kwargs': {'min_n_samples': 16000},
    },
    'model_oracle': {
        'rollout_kwargs': {'min_n_samples': 16000},
    },
}
configs_walker3d = copy.deepcopy(configs)
dict_update(configs_walker3d, t)
