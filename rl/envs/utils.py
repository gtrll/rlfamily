import gym
import os

from rl import envs
from rl.tools import supervised_learners as Sup

from pybullet_envs.minitaur.agents.tools import BatchEnv
from pybullet_envs.minitaur.agents.tools.wrappers import ExternalProcess


def create_batch_env(envid, seed, n_envs, render=False, use_ext_proc=True):
    def cons(): return create_env(envid, seed, render=False)
    func = cons if not use_ext_proc else lambda: ExternalProcess(cons)
    envs = [func() for _ in range(n_envs)]
    env = BatchEnv(envs, blocking=not use_ext_proc)
    return env


def create_env(envid, seed, render=False):
    """Create minitaur or other standard gym environment."""
    if 'minitaur' in envid:
        from pybullet_envs.minitaur.agents.scripts import utility
        from pybullet_envs.minitaur.agents import tools
        config = utility.load_config(os.path.expanduser('minitaur_config'))
        if 'bad' in envid:
            with config.unlocked:
                config.env.keywords['accurate_motor_model_enabled'] = False
                config.env.keywords['control_latency'] = .0
                config.env.keywords['pd_latency'] = .0
                config.env.keywords['urdf_version'] = None
        env = config.env(render=render)
        if config.max_length:
            env = tools.wrappers.LimitDuration(env, config.max_length)
        env = tools.wrappers.RangeNormalize(env)
        env = tools.wrappers.ClipAction(env)
        env = tools.wrappers.ConvertTo32Bit(env)

        class MySpec(object):
            def __init__(self, max_episode_steps):
                self.max_episode_steps = max_episode_steps

        env.spec = MySpec(1000)
    else:
        env = gym.make(envid)

    # Set up seed.
    env.seed(seed)
    return env


# Env.
ENVID2MODELENV = {
    'DartCartPole-v1': envs.Cartpole,
    'DartHopper-v1': envs.Hopper,
    'DartSnake7Link-v1': envs.Snake,
    'DartWalker3d-v1': envs.Walker3d,
}


def create_sim_env(env, seed, inaccuracy=None, dyn_configs=None):
    """ Create an EnvWithModel object as a model of env."""
    if dyn_configs:
        # Learnable dynamics.
        st = env.env.get_state() if hasattr(env, 'env') else env.get_state()
        st_dim, ac_dim = len(st), env.action_space.shape[0]
        dyn_cls = getattr(Sup, dyn_configs['dyn_cls'])
        # build_nor = Nor.create_build_nor_from_str(dyn_configs['nor_cls'], dyn_configs['nor_kwargs'])
        dyn = dyn_cls(st_dim + ac_dim, st_dim, **dyn_configs['dyn_kwargs'])
        predict = dyn.predict
    else:
        predict = None

    envid = env.env.spec.id
    sim_env = ENVID2MODELENV[envid](env, predict=predict, model_inacc=inaccuracy, seed=seed)
    return sim_env
