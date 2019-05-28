ranges_common = [
    [['general', 'seed'], [x * 100 for x in range(4)]],
    [['oracle', 'or_kwargs', 'use_log_loss'], [True]],
    [['oracle', 'or_kwargs', 'normalize_weighting'], [False]],
    [['algorithm', 'use_shift'], [False]],
    [['algorithm', 'reg_damping'], [0.1]],
]
ranges_mf = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['model-free']],
]

ranges_lazy1 = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['piccolo']],
    [['model_oracle', 'mor_cls'], ['LazyOracle']],
]

ranges_agg1 = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['piccolo']],
    [['model_oracle', 'mor_cls'], ['AggregatedOracle']],
    [['model_oracle', 'aggor_kwargs', 'max_n_iterations'], [5]],
]

ranges_sim1 = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['piccolo']],
    [['model_oracle', 'mor_cls'], ['SimulationOracle']],
    [['model_oracle', 'env_type'], ['true']],
]

ranges_sim1_models = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['piccolo']],
    [['model_oracle', 'mor_cls'], ['SimulationOracle']],
    [['model_oracle', 'env_type'], ['true', 0.2, 0.5, 0.8]],
]

ranges_dyna_sim1 = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['dyna']],
    [['model_oracle', 'mor_cls'], ['SimulationOracle']],
    [['model_oracle', 'env_type'], ['true']],
]

ranges_adv = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['piccolo']],
    [['model_oracle', 'mor_cls'], ['AdversarialOracle']],
]

ranges_dyna_adv = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['dyna']],
    [['model_oracle', 'mor_cls'], ['AdversarialOracle']],
]


############################
ranges_sim1_opt = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['piccolo']],
    [['model_oracle', 'mor_cls'], ['SimulationOracle']],
    [['model_oracle', 'env_type'], [0.2]],
    [['algorithm', 'n_model_steps'], [5]],
    [['algorithm', 'alg_kwargs', 'warm_start'], [True]],
    [['algorithm', 'alg_kwargs', 'update_in_pred'], [True]],
]

ranges_sim1_opt_models = [
    [['algorithm', 'alg_kwargs', 'update_rule'], ['piccolo']],
    [['model_oracle', 'mor_cls'], ['SimulationOracle']],
    [['model_oracle', 'env_type'], ['true', 0.2, 0.5, 0.8]],
    [['algorithm', 'n_model_steps'], [5]],
    [['algorithm', 'alg_kwargs', 'warm_start'], [True]],
    [['algorithm', 'alg_kwargs', 'update_in_pred'], [True]],
]


STEPSIZE = {
    'cp': {'etas': {'adam': 0.005, 'natgrad': 0.05, 'trpo': 0.002},
           'c': 0.1},
    'hopper': {'etas': {'adam': 0.005, 'natgrad': 0.05, 'trpo': 0.002},
               'c': 0.1},
    'snake': {'etas': {'adam': 0.002, 'natgrad': 0.2, 'trpo': 0.01},
              'c': 0.1},
    'walker3d':  {'etas': {'adam': 0.01, 'natgrad': 0.2, 'trpo': 0.04},
                  'c': 0.01},
}


def get_ranges(env, ranges_name, base_algorithms):
    """Combines ranges_common, ranges_env_specific, and ranges_algorithm_specific.

    When ranges_name ends with a positive integer, it specifies the number of iterations, e.g. mf_300.
    """
    etas = STEPSIZE[env]['etas']
    c = STEPSIZE[env]['c']
    etas = [etas[alg] for alg in base_algorithms]  # get the etas that are needed
    ranges_env_specific = [
        [['algorithm', 'base_alg'], base_algorithms, ['algorithm', 'learning_rate', 'eta'], etas],
        [['algorithm', 'learning_rate', 'c'], [c]],
    ]
    split = ranges_name.split('_')
    if split[-1].isdigit():
        ranges_env_specific += [
            [['experimenter', 'run_alg_kwargs', 'n_itrs'], [int(split[-1])]],
        ]
        ranges_name = '_'.join(split[:-1])

    ranges = ranges_common + ranges_env_specific + globals()['ranges_' + ranges_name]

    return ranges
