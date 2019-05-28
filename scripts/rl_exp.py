import argparse
import tensorflow as tf

from scripts import configs as C
from rl import experimenter as Exp
from rl.configs import parser as ps
from rl.tools.utils import tf_utils as U


def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create env and fix randomness
    env, envid, seed = ps.general_setup(c['general'])

    # Create objects for defining the algorithm
    policy = ps.create_policy(env, seed, c['policy'])
    ae = ps.create_advantage_estimator(policy, c['advantage_estimator'])
    oracle = ps.create_oracle(policy, ae, c['oracle'])
    model_oracle = ps.create_model_oracle(oracle, env, envid, seed, c['model_oracle'])

    # Enter session.
    U.single_threaded_session().__enter__()
    tf.global_variables_initializer().run()

    # Create algorithm after initializing tensors, since it needs to read the
    # initial value of the policy.
    alg = ps.create_algorithm(policy, oracle, c['algorithm'], model_oracle=model_oracle)

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, env, c['experimenter']['rollout_kwargs'])
    exp.run_alg(**c['experimenter']['run_alg_kwargs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs_name', type=str)
    args = parser.parse_args()
    configs = getattr(C, args.configs_name)
    main(configs)
