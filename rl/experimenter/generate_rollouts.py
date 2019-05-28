import time
from rl.experimenter.rollout import RO, Rollout


def generate_rollout(pi, logp, env,
                     min_n_samples, max_n_rollouts, max_rollout_len,
                     with_animation=False):
    """
    Collect rollouts until we have enough samples. All rollouts are COMPLETE in
    that they never end prematurely: they end either when done is true or
    max_rollout_len is reached.

    Args:
        pi: a function that maps ob to ac
        logp is either None or a function that maps (obs, acs) to log probabilities
        max_rollout_len: maximal length of a rollout
        min_n_samples: minimal number of samples to collect
        max_n_rollouts: maximal number of rollouts
        with_animation: display animiation of the first rollout
    """
    n_samples = 0
    rollouts = []

    if max_rollout_len is None:
        max_rollout_len = env._max_episode_steps
    else:
        max_rollout_len = min(env._max_episode_steps, max_rollout_len)

    def get_state():
        if hasattr(env, 'env'):
            return env.env.state  # openai gym env, which is a TimeLimit object
        else:
            return env.state

    while True:
        animate_this_rollout = len(rollouts) == 0 and with_animation
        ob = env.reset()
        st = get_state()
        obs, acs, rws, sts = [], [], [], []
        steps = 0  # steps so far
        while True:
            if animate_this_rollout:
                env.render()
                time.sleep(0.05)
            obs.append(ob)
            sts.append(st)
            ac = pi(ob)
            acs.append(ac)
            ob, rw, done, _ = env.step(ac)
            st = get_state()
            rws.append(rw)
            steps += 1
            if max_rollout_len is not None and steps >= max_rollout_len:
                # breaks due to steps limit
                sts.append(st)
                obs.append(ob)
                absorb = False
                break
            elif done:
                # breaks due to absorbing state
                sts.append(st)
                obs.append(ob)
                absorb = True
                break

        rollout = Rollout(obs, acs, rws, sts, absorb, logp)
        rollouts.append(rollout)
        n_samples += len(rollout)
        if ((min_n_samples is not None and n_samples >= min_n_samples) or
                (max_n_rollouts is not None and len(rollouts) >= max_n_rollouts)):
            break
    ro = RO(rollouts)
    return ro
