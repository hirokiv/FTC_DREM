import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate


class FTC_DREM_Estimator:


    def __init__(self, initial_theta, gamma, mu=0.98, TC=np.infty, initial_time=0, TD=0.2, mode='FTC-D', delta_mode='PE', sigma_o=0.0):
        """
        Initialize the FTC DREM estimator.
        :param initial_theta: Initial parameter estimates.
        :param gamma: Learning rate.
        :param mu: Upper bound for the clipping function, adjusting convergence behavior.
        :param TC: Critical time after which convergence is assumed to be achieved.
        :param initial_time: Starting time of the simulation for tracking initial conditions.
        """
        self.initial_theta = np.array(initial_theta, dtype=float)
        self.theta_hat = np.array(initial_theta, dtype=float)
        # self.theta_hat = initial_theta
        self.gamma = gamma
        self.mu = mu
        self.TC = TC
        self.TD = TD
        self.t0 = initial_time
        self.mode = mode
        self.w = 1.0  # Initialize w at t0 to be clipped value of 1
        self.delta_mode = delta_mode
        self.sigma_o = sigma_o
        np.random.seed(42)

    def delta(self, t):
        """
        Simulated Delta(t) function reflecting system responsiveness or other dynamics.
        """
        if self.delta_mode == 'PE':
            return np.sin( 2 * np.pi * (t))  # Just a placeholder for actual dynamics
        elif self.delta_mode == 'not_in_L2':
            return 1 / (t + 1)  # Just a placeholder for actual dynamics

        else:
            return 0
        #return 1/(t+1)  # Just a placeholder for actual dynamics

    def delta_obs(self, t):
        delta = self.delta(t)
        delta_obs = delta + np.random.normal(0, self.sigma_o)
        return delta_obs

    def compute_y(self, t, obs=False):
        # true system
        if obs==False:
            y = self.delta(t) * self.true_parameters(t)
        elif obs==True:
            y = self.delta_obs(t) * self.true_parameters(t)
        return y


    def dynamics(self, t, input):
        """
        Differential equation that models the dynamics of the estimator.
        """
        theta, w = input

        # true_theta = self.true_parameters(t)
        delta_t_obs = self.delta_obs(t)

        y_obs = self.compute_y(t, obs=True) 
        dtheta_dt = self.gamma * delta_t_obs * (y_obs - delta_t_obs * theta)

        if self.mode=='LS':
            dw_dt = 0

        elif self.mode=='FTC':
            dw_dt = - self.gamma * delta_t_obs * delta_t_obs * w

        elif self.mode=='FTC-D':
            if t < self.TD:
                delta_t_TD  = 0
            else:
                delta_t_TD  = self.delta_obs(t - self.TD)

            delta_t2    = delta_t_obs    * delta_t_obs
            delta_t_TD2 = delta_t_TD * delta_t_TD

            dw_dt       = - self.gamma * (delta_t2 - delta_t_TD2) * w
        else:
            return 0

        return self.concat_y(dtheta_dt, dw_dt)

    def interp_theta_TD(self, tlist, theta):
        # compute theta(t-TD)
        t_TD = tlist - self.TD
        f = interpolate.interp1d(tlist, theta, fill_value=(self.initial_theta,np.nan), bounds_error=False)
        theta_TD = f(t_TD)
        return theta_TD


    def compute_FTC_Est(self, tlist, theta, w):
        # final form computed for FTC and FTC-D in Eq. 33 and Eq 35
        if self.mode == 'FTC-D':
            wd = self.compute_wc(0, w)
            return 1/(1-wd) * (theta - wd * self.interp_theta_TD(tlist, theta) )
            #return 1/(1-wd) * (theta - wd * self.initial_theta )

        elif self.mode == 'FTC':
            wc = self.compute_wc(0, w)
            return 1/(1-wc) * (theta - wc * self.initial_theta)
        

    def true_parameters(self, t):
        """
        Function to simulate time-varying true parameters.
        """
        # As described earlier, depending on the application scenario
        if t < 10:
            return 10
        elif 10 <= t < 20:
            return 15
        elif 20 <= t < 30:
            return 15 - 0.5 * (t - 20)
        else:
            return 10

    def true_parameters_list(self, tlist):
        """
        Return theta for tlist span
        """
        theta_t = np.zeros(len(tlist))
        for idx, t in enumerate(tlist):
            theta_t[idx] = self.true_parameters(t)
        return theta_t


    def compute_wc(self, t, w):
        """
        Compute the weight function w_c(t) based on current time and system dynamics.
        """
        if t <= self.TC:
            return self.clip(w)
        return self.clip(0)  # After TC, we assume the parameters have converged.

    def clip(self, value):
        """
        Clip the value between 0 and mu to ensure stability and convergence.
        """
        return np.clip(value, None, self.mu)

    def solve(self, t_span, t_eval=None):
        """
        Solve the differential equation over the specified time span.
        """
        sol = solve_ivp(self.dynamics, t_span, self.concat_y(self.initial_theta, self.w),  method='RK45', dense_output=True, t_eval=t_eval, atol=1e-6, rtol=1e-6)
        return sol.t, sol.y[0], sol.y[1]

    def concat_y(self, theta_hat, w):

        if np.isscalar(w):
            w = [w]

        if np.isscalar(theta_hat):
            theta_hat = [theta_hat]

        return np.concatenate([theta_hat, w])
        