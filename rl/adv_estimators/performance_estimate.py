import numpy as np


class PerformanceEstimate(object):
    """
      A helper function to compute gradient of the form
          E_{d_\pi} (\nabla E_{\pi}) [ A ]
      where
          the unnormalized state distribution is
              d_{\pi} = \sum_{t=1}^\infty  \gamma^t * d_{pi,t}
          and for \lambda in [0,1],
              A_t = (1-\lambda)  \sum_{k=0}^\infty \lambda^k  A_{k,t}
          with
              A_{k,t} = c_t - v_t + \delta * V_{k,t+1}
              V_{k,t} = w_t * c_t + \delta * w_t * w_{t+1} * c_{t+1} + ...
                        + \delta^{k-1} * w_t * ... * w_{t+k-1} * c_{t+k-1}
                        + \delta^k     * w_t * ... * w_{t+k-1} * v_{t+k}
              c is the instantaneous cost,
              w is the importance weight
              v is the baseline

          In implementationn, A_t is computed as
              A_t = x_t + (\lambda*\delta) * X_{t+1} + Y_t

              where x_t =     c_t - v_t + \delta *    v_{t+1}
                    X_t = (w*c)_t - v_t + \delta * (wv)_{t+1}
                    Y_t = \sum_{k=2}^\infty (\lambda*\delta)^k * w_{t+1} * ... * w_{t+k-1} X_{t+k}

      =========================================================================================
      \gamma in [0,1] is the discount factor in the problem
      \lambda in [0,1] defines the \lambda-mixing of a family of estimates
      \delta in [0, \gamma], w, and V define the reshaped costs

      The feature of the estimator is determined by the following criteria:

      1) \delta==\gamma or \delta<\gamma:
          whether to use the same discount factor as the problem's definition
          for estimating value function. Using smaller delta simplifes the
          estimation problem but introduces additional bias.

      2) \lambda==1 or  \lambda <1:
          whether to use Monte-Carlo rollouts or a \lambda-weighted estimate,
          which has larger bias but smaller vairance.

      3) w==1, or w==p(\pi*)/p(\pi):
          whether to use importance sampling to estimate the advantage function
          with respect to some other policy \pi* using the samples from the
          exploration policy \pi. The use of non-identity w can let A_{V, k}
          to estimate the advantage function with respect to \pi* even when the
          rollouts are collected by \pi.


      Some examples (and their imposed constraints):

      1) Actor-Critic Family (\delta==\gamma) (w=1)
          a) \lambda==1, unbiased Monte-Carlo rollout with costs reshaped by some
                      arbitrary function V
          b) \lambda==0, basic Actor-Critic when V is the current value estimate
          c) \labmda in (0,1), lambda-weighted Actor-Critic, when V is the current
                      value estimate.

      2) GAE Family (\delta<\gamma) (w=1)
          a) \gamma==1, (\delta, \lambda)-GAE estimator for undiscounted problems
                     when V is the current value estimate (Schulmann et al.,
                     2016)
          b) \gamma in (0,1], (\delta, \lambda)-GAE for \gamma-discounted
                     problems, when V is the current value estimate

      3) PDE (Performance Difference Estimate) Family (w = p(\pi') / p(\pi) ):
          PDE builds an estimate of E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]
          where A_{\pi'} is the (dis)advantage function wrt \pi', in which
          V is the value estimate of some arbitrary policy \pi', \lambda in [0,
          1] and \delta in [0, \gamma] are bias-variance.
    """

    def __init__(self, gamma, lambd=0., delta=None, default_v=0.0):
        delta = np.min([delta, gamma]) if delta is not None else np.min([gamma, 0.9999])
        self.gamma = np.clip(gamma, 0., 1.)
        self.delta = np.clip(delta, 0., 1.)
        self.lambd = np.clip(lambd, 0., 1.)
        self.default_v = default_v  # the value function of absorbing states

    @staticmethod
    def shift_l(v, padding=0.):
        return np.append(v[1:], padding) if np.array(v).size > 1 else v

    def reshape_cost(self, c, V, done, w=1., padding=0.):
        v, v_next = V[:-1], V[1:]
        if done:  # ignore the last element
            v_next[-1] = padding
        return w * (c + self.delta * v_next) - v

    def dynamic_program(self, a, b, c, d, w):
        # Compute the expression below recursibely from the end
        #   val_t = d^t * ( a_t + \sum_{k=1}^infty c^k w_{t+1} ... w_{t+k} b_{t+k} )
        #         = d^t * ( a_t + e_t )
        #
        # in which e_t is computed recursively from the end as
        #
        #   e_t = \sum_{k=1}^infty c^k w_{t+1} ... w_{t+k} b_{t+k} )
        #       = c w_{t+1} b_{t+1} + \sum_{k=2}^infty c^k w_{t+1} ... w_{t+k} b_{t+k} )
        #       = c w_{t+1} b_{t+1} + c w_{t+1} \sum_{k=1}^infty c^k w_{t+1+1} ... w_{t+1+k} b_{t+1+k} )
        #       = c w_{t+1} b_{t+1} + c w_{t+1} e_{t+1}
        #       = c w_{t+1} (b_{t+1} + e_{t+1})
        #
        # where the boundary condition of e is zero.

        assert len(a) == len(b), 'Lengths of the two sequences do not match.'
        horizon = len(a)
        if type(w) is not np.ndarray:
            w = np.full_like(a, w)  # try to make it one
        e = np.zeros_like(a)  # last val is 0
        cw = c * w
        for i in reversed(range(1, len(e))):
            e[i - 1] = cw[i] * (b[i] + e[i])
        val = (d**np.arange(horizon)) * (a + e)
        return val

    def adv(self, c, V, done, w=1., lambd=None, gamma=None):
        # compute pde such that
        #     \sum_{s,a in traj_pi} \nabla \log p(a|s) pde(s,a)
        # is unbiased estimate of
        #     E_{d_\pi} (\nabla E_{\pi}) [ A_V ]
        #

        # V is an np.ndarray with length equal to len(c)+1
        # w can be int, float, or np.ndarray with length equal to len(c)
        # if done is True, the last element of V is set to the default value

        assert len(c) + 1 == len(V), 'V needs to be one element longer than c.'
        assert type(done) is bool
        gamma = gamma if gamma is not None else self.gamma
        lambd = lambd if lambd is not None else self.lambd

        X = self.reshape_cost(c, V, done, w=w, padding=self.default_v)
        x = self.reshape_cost(c, V, done, w=1.0, padding=self.default_v)
        Y = self.shift_l(X) * self.delta * lambd  # this always pads 0

        a = x + Y
        b = Y
        c = lambd * self.delta
        d = gamma
        return self.dynamic_program(a, b, c, d, w)

    def qfn(self, c, V, done, w=1., lambd=None, gamma=None):
        # compute the qfn such that
        #     \sum_{s,a in traj_pi} qfn(s,a)
        # is unbiased estimate of
        #     A_V + V
        # under policy pi
        #   w is the importance sampling weight
        return V + self.adv(c, V, done, w, lambd, gamma)
