import numpy as np
import scipy
from scipy.optimize import minimize, LinearConstraint
import polytope
import pypoman

# from https://github.com/GitPascalP/masc/blob/main/masc/mpc/controller.py

def safe_costs(u, u_rl):
    """ cost function for safeguard """
    cost = np.abs((u - u_rl))**2
    return cost


def safe_costs_derivative(u, u_rl):
    """ derivative of the quadratic cost function """
    derivative = 2 * (u - u_rl)
    return derivative


class Safeguard:
    """ safeguard to ensure that the RL-Agent takes only safe actions """

    def __init__(self, constraints, set_scaler=None, savepath=None, state_limits = None, action_limits = None):
        """
        args:
            constraints (polytope): polytope describing the feasible set
            set_scaler (float): scaling factor to manually scaler the
                                constraint-polytope
            savepath (str): estimated polytopes are saved to path during
                            learning if not None
        """
        # attributes, that have to be updated during process
        self.constraints_F0 = constraints
        self.guard_constraints = None
        self.set_scaler = set_scaler or 0.0 # todo: what to do with that?
        self.delta_action = 1 # change in worst case (if not safe) action by 1V into direction
        self.state_lim = state_limits
        self.action_lim = action_limits
        self.sol_not_found = 0
        self.rand_sol_not_found = 0

        if self.constraints_F0 is not None:
            self.update(
                constraints=self.constraints_F0,
                mode='update',
            )

    def scale(self, constraints, scaler):
        """ scale constraint polytope """
        if constraints.A.shape[-1] == 2:
            weights = np.array(
                [1 + self.set_scaler + np.squeeze(scaler[0]),
                 1 + self.set_scaler + np.squeeze(scaler[1])])
            weights = np.squeeze(weights)
        elif constraints.A.shape[-1] == 3:
            weights = np.array(
                [1 + self.set_scaler + np.squeeze(scaler[0]),
                 1 + self.set_scaler + np.squeeze(scaler[1]),
                 1])
            weights = np.squeeze(weights)

        scaled_constraints = polytope.Polytope(
            constraints.A * weights, constraints.b
        )
        return scaled_constraints

    def update(
            self,
            constraints=None,
            constraints2d=None,
            fit_error_margin=np.array([0, 0]),
            mode='adapt',
    ):
        """
        update Safeguard constraints during learning process, by scaling
        the polytope with different coefficients
        """

        if mode == 'adapt':
            self.guard_constraints = self.scale(
                self.constraints_F0, scaler=fit_error_margin
            )
        elif mode == 'update':
            # set new feasible set given in arguments
            self.guard_constraints = self.scale(
                constraints, scaler=fit_error_margin
            )

    def guide(self, action, state):
        """ guide the action to keep constraints and solve QP if necessary """

        active = False

        state = np.atleast_1d(np.squeeze(state))
        action = np.atleast_1d(np.squeeze(action))
        n, m = state.shape[0], action.shape[0]

        # Check - if state is out of bounds: easiest approach: assume clipping at limits to choose valid action
        # todo: include barrier here to bring state back to valid range and choose action appropriate
        state[0] = np.clip(state[0], -self.state_lim[0], self.state_lim[0])
        state[1] = np.clip(state[1], -self.state_lim[1], self.state_lim[1])

        if np.abs(action) > self.action_lim:
            active = True
            action = np.clip(action, -self.action_lim, self.action_lim)

        state_act = np.concatenate([state, action])

        if self.check_constraints(state_act) and not active:
            # rl action is safeguard is inactive
            sg_active = False
            u_safe = action

        elif self.check_constraints(state_act) and active:
            # save via clipping
            sg_active = True
            u_safe = action

        else:
            # rl actions will violate constraints and guard is actuated
            sg_active = True

            # construct constraint for scipy solver
            Fg, Fe = self.guard_constraints.A, self.guard_constraints.b
            Fg_u = Fg[:, -1]
            Fe_u = Fe - (Fg[:, :n] @ state)
            Fe_u_l = -np.ones(Fe_u.shape)*400 #* (-np.inf)
            #print("lower bound for finding safe action is set to -400V!")
            Fg_u = Fg_u.reshape(1, -1).T


            # compared to https://stackoverflow.com/questions/52001922/linearconstraint-in-scipy-optimize
            ineq_cons = {'type': 'ineq',
                         'fun': lambda action: (Fe - (
                                     Fg[:, :n] @ state)) - Fg_u @ action,
                         }
            lin_constraints = LinearConstraint(A=Fg_u, lb=Fe_u_l, ub=Fe_u)
            # solve constrained opt-problem to find safe action

            solution = scipy.optimize.minimize(
                safe_costs,
                x0=np.array([0]),
                args=(action),
                #constraints=lin_constraints,
                constraints=ineq_cons,
                method='COBYLA',
                #method='Nelder-Mead',
                #bounds=[(-self.action_lim, self.action_lim)]
            )


            u_safe = solution.x
            if not solution.success:
                # clip action to action limits (-1, 1)
                u_safe = np.clip(u_safe, -400, 400)
                self.sol_not_found = 1

            if not self.check_constraints(np.concatenate([state, u_safe])):
                # Since sometimes it is aprrox the boarder go one step into direction 3 times
                # vertices = pypoman.compute_polytope_vertices(self.guard_constraints.A, self.guard_constraints.b)
                direction = np.sign(u_safe - action) # assume direction of u_Safe is correct
                count = 0
                while not self.check_constraints(np.concatenate([state, u_safe])):
                    # todo: no while for real application - just for sim testing
                    u_safe = u_safe + direction * self.delta_action
                    count +=1
                    if count > 3:
                        # print('no safe action found!')
                        break

                count = 0
                while not self.check_constraints(np.concatenate([state, u_safe])):
                    # if solution is not found at all, chosse random action which is in polytope
                    u_safe = np.random.uniform(-self.action_lim, self.action_lim)
                    if count > 10:
                        self.rand_sol_not_found = 1
                        break


                # self.check_constraints(np.concatenate([np.array([183.25, 38.51]), np.array([-0.75*400])]))

        return u_safe, sg_active

    def check_constraints(self, state_action):
        """ check if state-action vector lies within feasible set """
        return state_action in self.guard_constraints