import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, List

import sys
sys.path.append("./python-filter")
import rudolfpy as rd

class ValidationModel:
    """
    Class for validating sensor tasking performance

    Args:
        u (np.ndarray[int]): tasking schedule / control

    """
    def __init__(self,
                 u: np.ndarray[int],
                 groundtruth_observers: np.ndarray[float],
                 groundtruth_targets: np.ndarray[float],
                 estimate_observers: np.ndarray[float],
                 estimate_targets: np.ndarray[float],
                 P0_observers: np.ndarray[float],
                 P0_targets: np.ndarray[float],
                 filter , #TODO  Base Filter type hint
                 timestep: float):

        self.u = u
        self.groundtruth_observers = groundtruth_observers
        self.groundtruth_targets = groundtruth_targets
        self.estimate_observers = estimate_observers
        self.estimate_targets = estimate_targets
        self.P0_observers = P0_observers
        self.P0_targets = P0_targets
        self.filter = filter
        self.timestep = timestep

        match self.filter.measurement_model.name:
            case "Angle":
                self.func_simulate_measurements = rd.func_simulate_measurement_angle
            case "Angle_AngleRate":
                self.func_simulate_measurements = rd.func_simulate_measurement_angle_anglerate
            case "PositionVector":
                self.func_simulate_measurements = rd.func_simulate_measurements
            case "BaseMeasurement":
                raise ValueError("Must use a derived class of BaseMeasurement")
            case _:
                raise NotImplementedError("This measurement model is not supported")
        
    

    def _get_t_measurements_target(self, target_idx: int) -> tuple[np.ndarray[float], np.ndarray[int]]:
        """
        Get a list of measurement times for the specified target, according to the control

        Args:
            target_idx (int): index of the target to get measurements for

        Returns:
            t_measurements (np.ndarray[float]): a vector of measurement times for the target
            observer_indices (np.ndarray[int]): a vector of observer indices at the relevant times for the target
        """
        u_j = self.u[:, target_idx, :]
        # num_steps = u_j.shape[1]
        # num_observers = u_j.shape[0]

        # t_measurements = []
        # observer_indices = []

        # for k in range(num_steps):
        #     t_meas = self.timestep * (k + 0.5)
        #     for i in range(num_observers):
        #         if u_j[i, k] == 1:
        #             t_measurements.append(t_meas)
        #             observer_indices.append(i)

        # t_measurements = np.array(t_measurements, dtype=float)
        # observer_indices = np.array(observer_indices, dtype=int)

        indices = np.where(u_j == 1)

        t_measurements = self.timestep * (indices[1] + 0.5)
        observer_indices = indices[0]

        return t_measurements, observer_indices
    
    def _get_measurement_info_target(self, target_idx: int, filter_observers: Optional[bool] = False) -> tuple[np.ndarray, list]:
        """
        Get a measurement times and list of parameters for target measurement. Use default values.

        Args:
            target_idx (int): index of chosen target        
        Returns:
            t_measurements (np.ndarray[float]): vector of measurement times
            params_measurements (list): list of measurement parameters for specified target
        """


        match self.filter.measurement_model.name:
            case "Angle":
                sigma_phi = np.deg2rad(0.5) # std of 0.5 degrees in angle measurement
                t_measurements, observer_indices = self._get_t_measurements_target(target_idx=target_idx)
                if filter_observers:
                    # TODO 
                    raise NotImplementedError
                else:
                    observer_histories = {}
                    for unique_observer_idx in np.unique(observer_indices):
                        observer_histories[unique_observer_idx] = self._get_observer_history_groundtruth(observer_idx=unique_observer_idx, t_eval=t_measurements)

                r_observers = []

                for i in range(len(t_measurements)):
                    observer_idx = observer_indices[i]
                    observer_state, _ = observer_histories[observer_idx][i]
                    r_observers.append(observer_state[:3])

                params_measurements = [[r_observer, sigma_phi] for r_observer in r_observers ]
                
            case "Angle_AngleRate":
                sigma_phi = np.deg2rad(0.5) # std of 0.5 degrees in angle measurement
                dt = 3600 / self.filter.dynamics.TU # one hour exposure time for measurement

                t_measurements, observer_indices = self._get_t_measurements_target(target_idx=target_idx)
                if filter_observers:
                    # TODO 
                    raise NotImplementedError
                
                else:
                    observer_histories = {}
                    for unique_observer_idx in np.unique(observer_indices):
                        observer_histories[unique_observer_idx] = self._get_observer_history_groundtruth(observer_idx=unique_observer_idx, t_eval=t_measurements)

                x_observers = []

                for i in range(len(t_measurements)):
                    observer_idx = observer_indices[i]
                    observer_state, _ = observer_histories[observer_idx][i]
                    x_observers.append(observer_state)

                params_measurements = [[x_observer, sigma_phi, dt] for x_observer in x_observers ]


            case "PositionVector":
                # TODO
                raise NotImplementedError
            
        return t_measurements, params_measurements


    
    def _get_t_measurements_observer(self, observer_idx: int, num_measurements: int) -> np.ndarray[float]:
        """
        Get a list of measurement times for the specified observer. Assume that the requested number of measurements is 
        equally distributed over the time span specified by the control.

        Args:
            observer_idx (int): index of the observers to get measurements for

        Returns:
            t_measurements (np.ndarray[float64]): a vector of measurement times for the observer
        """

        num_steps = self.u.shape[2]
        tspan = (0, self.timestep * num_steps)

        t_measurements = np.linspace(tspan[0], tspan[1], num_measurements)

        return t_measurements
    
    def _get_measurement_info_observer(self, observer_idx: int) -> tuple[np.ndarray, list]:
        """
        Get measurement times and list of parameters for observer measurement. Use default values.

        Args:
            observer_idx (int): index of chosen observer        
        Returns:
            t_measurements (np.ndarray[float]): vector of measurement times
            params_measurements (list): list of measurement parameters for specified observer
        """

        match self.filter.measurement_model.name:
            case "Angle":
                sigma_phi = np.deg2rad(0.5) # std of 0.5 degrees in angle measurement
                t_measurements = self._get_t_measurements_observer(observer_idx=observer_idx)
                r_observer = np.array([1.00, 0, 0])  # TODO fix hardcoded static position of the entity taking the measurement
                params_measurements = [[r_observer, sigma_phi] for _ in range(len(t_measurements))]

            case "Angle_AngleRate":
                sigma_phi = np.deg2rad(0.5) # std of 0.5 degrees in angle measurement
                dt = 3600 / self.filter.dynamics.TU # one hour exposure time for measurement

                t_measurements = self._get_t_measurements_observer(observer_idx=observer_idx)
                x_observer = np.array([1.00, 0, 0, 0, 0, 0])  #TODO fix hardcoded static position and velocity of the entity taking the measurement
                params_measurements = [[x_observer, sigma_phi, dt] for _ in range(len(t_measurements))]

            case "PositionVector":
                # TODO
                raise NotImplementedError

        return t_measurements, params_measurements
    
    def _get_observer_history_groundtruth(self,
                                          observer_idx:int,
                                          t_eval: Optional[np.ndarray] = None) -> list[tuple]:
        """
        Get observer state history, using the ground truth observer initial condition

        Args:
            observer_idx (int): index of the observer to propagate
            t_eval (np.ndarray[float]): vector of times to evaluate solution at

        Returns:
            observer_history (list[tuple]): observer history as a list of (x, P) tuples.
        """
        if t_eval is None:
            num_steps = self.u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        x0 = self.groundtruth_observers[observer_idx]

        sol_true = self.filter.dynamics.solve([0.0, t_eval[-1]], x0, t_eval=t_eval)

        observer_history = [(sol_true.y[0:6, idx], None) for idx in range(t_eval.size)]

        return observer_history

    def _get_target_history_groundtruth(self,
                                        target_idx:int,
                                        t_eval: Optional[np.ndarray] = None) -> list[tuple]:
        """
        Get target state history, using the ground truth target initial condition

        Args:
            target_idx (int): index of the target to propagate
            t_eval (np.ndarray[float]): vector of times to evaluate solution at

        Returns:
            sol_true (OdeResult): returned object from `scipy.integrate.solve_ivp`
        """
        if t_eval is None:
            num_steps = self.u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        x0 = self.groundtruth_targets[target_idx]

        sol_true = self.filter.dynamics.solve([0.0, t_eval[-1]], x0, t_eval=t_eval)

        target_history = [(sol_true.y[0:6, idx], None) for idx in range(t_eval.size)]

        return target_history
    
    def _get_observer_history_filter(self,
                                     observer_idx: int,
                                     params_measurements: list,
                                     t_measurements: np.ndarray[float],
                                     t_eval: Optional[np.ndarray[float]] = None) -> list[tuple]:
        """
        Get the observer state history, using a filter to estimate the observer position

        Args:
            observer_idx (int): index of the observer to propagate
            params_measurements (list): list of measurement parameters to pass to `func_simulate_measurements` for each measurement taken.
            t_measurements (np.ndarray[float]): vector of times when measurements of observer state are taken
            t_eval (np.ndarray[float]): vector of times to query the filtered state at

        Returns:
            observer_history (list[tuple[np.ndarray, np.ndarray]]): state history of observer.

        Notes:
            - params_measurements must have same length as t_measurements
        """

        if len(params_measurements) != t_measurements.size:
            raise AssertionError(f"params_measurements (size: {len(params_measurements)}) must be of same size as t_measurements (size: f{t_measurements.size}).")

        if t_eval is None:
            num_steps = self.u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        observer_history_gt = self._get_observer_history_groundtruth(observer_idx=observer_idx, t_eval=t_measurements)
        observer_history = []

        x0 = self.estimate_observers[observer_idx]
        P0 = self.P0_observers[observer_idx]

        # initialize filter
        self.filter.t = 0
        self.filter.x = x0
        self.filter.P = P0

        # do estimation
        count_m = 0
        count_e = 0
        M = t_measurements.size
        E = t_eval.size
        while count_m < M or count_e < E:
            # compare first two values of two lists, pop the minimum and return whether it is measurement or evaluation
            next_time, type = _get_history_filter_helper(t_measurements, t_eval, count_m, count_e)
            if next_time > self.filter.t:
                _ = self.filter.predict(tspan = [self.filter.t, next_time])

            match type:
                case "eval":
                    x = self.filter.x
                    P = self.filter.P
                    observer_history.append((deepcopy(x), deepcopy(P)))
                    count_e += 1
                case "measure":
                    x_true = observer_history_gt[count_m][0]
                    y, R = self.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m][0])
                    count_m += 1
                case "both":
                    x_true = observer_history_gt[count_m][0]
                    y, R = self.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m][0])
                    x = self.filter.x
                    P = self.filter.P
                    observer_history.append((deepcopy(x), deepcopy(P)))
                    count_m += 1
                    count_e += 1

        return observer_history
    
    def _get_target_history_filter(self,
                                   target_idx: int,
                                   params_measurements: list,
                                   t_measurements: Optional[np.ndarray[float]] = None,
                                   t_eval: Optional[np.ndarray[float]] = None,
                                   filter_observers: Optional[bool] = False) -> list[tuple]:
        """
        Get the target state history at the requested times with the requested measurements

        Args:
            target_idx (int): index of the target to propagate
            params_measurements (list): list of measurement parameters to pass to `func_simulate_measurements` for each measurement taken.
            t_measurements (np.ndarray[float]): vector of times when measurements of target state are taken. If None, defaults to times by control
            t_eval (np.ndarray[float]): vector of times to query the filtered state at. If None, equally distributed amongst time horizon specified in control
            filter_observers (bool): Whether to filter observers. Defaults to False.

        Returns:
            target_history (list[tuple[np.ndarray, np.ndarray]]): state history of target.

        Notes:
            - if t_measurements is given, params_measurements must have same length as t_measurements. Otherwise, params_measurements is ignored.
        """

        if t_measurements is None:
            t_measurements, params_measurements = self._get_measurement_info_target(target_idx=target_idx, filter_observers=filter_observers)


        if len(params_measurements) != t_measurements.size:
            raise AssertionError(f"params_measurements (size: {len(params_measurements)}) must be of same size as t_measurements (size: f{t_measurements.size}).")
        

        if t_eval is None:
            num_steps = self.u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        target_history_gt = self._get_target_history_groundtruth(target_idx=target_idx, t_eval=t_measurements)
        target_history = []

        x0 = self.estimate_targets[target_idx]
        P0 = self.P0_targets[target_idx]

        # initialize filter
        self.filter.t = 0
        self.filter.x = x0
        self.filter.P = P0

        # do estimation
        count_m = 0
        count_e = 0
        M = t_measurements.size
        E = t_eval.size
        while count_m < M or count_e < E:
            # compare first two values of two lists, pop the minimum and return whether it is measurement or evaluation
            next_time, type = _get_history_filter_helper(t_measurements, t_eval, count_m, count_e)
            if next_time > self.filter.t:
                _ = self.filter.predict(tspan = [self.filter.t, next_time])

            match type:
                case "eval":
                    x = self.filter.x
                    P = self.filter.P
                    target_history.append((deepcopy(x), deepcopy(P)))
                    count_e += 1
                case "measure":
                    x_true = target_history_gt[count_m][0]
                    y, R = self.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m][0])
                    count_m += 1
                case "both":
                    x_true = target_history_gt[count_m][0]
                    y, R = self.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m][0])
                    x = self.filter.x
                    P = self.filter.P
                    target_history.append((deepcopy(x), deepcopy(P)))
                    count_m += 1
                    count_e += 1


        return target_history
    
    def run(self, filter_observers: Optional[bool] = False) -> Tuple[List[Tuple[np.ndarray[float], np.ndarray[float]]],
                                                                     List[Tuple[np.ndarray[float], np.ndarray[float]]]]: 
        """
        Run the validation model, filtering targets (and observers if necessary). Record history of target and observer states

        Args:
            filter_observers (bool): whether or not to use filtering for observer states

        Returns:
            Tuple(List, List): Tuple containing two lists:
                1. List of target histories. Each element is a tuple (x, P) of target state and covariance.
                2. List of observer histories. Each element is a tuple (x, P) of observer state and covariance.
        """

        num_targets = self.groundtruth_targets.shape[0]
        num_observers = self.groundtruth_observers.shape[0]

        target_histories = []
        observer_histories = []

        for i in range(num_targets):
            target_history = self._get_target_history_filter(target_idx=i, 
                                                            params_measurements=None, # will be ignored since t_measurements is None
                                                            t_measurements=None,  # default to measurements provided by control
                                                            t_eval=None, # default to 300 equally spaced points in the time horizon
                                                            filter_observers=filter_observers)
            target_histories.append(target_history)

        if filter_observers:
            # for j in range(num_observers):
                # observer_history = self._get_observer_history_filter(observer_idx=j, 
                #                                                     params_measurements=self.observer_measurement_params[j], 
                #                                                     t_measurements=self.observer_measurement_times[j], 
                #                                                     t_eval=self.observer_eval_times[j])
                # observer_histories.append(observer_history)
            raise NotImplementedError


        else:
            for j in range(num_observers):
                observer_history = self._get_observer_history_groundtruth(observer_idx=j,
                                                                          t_eval=None)
                
                observer_histories.append(observer_history)
                

        
        return target_history, observer_history
    

def _get_history_filter_helper(t_measurements: np.ndarray[float],
                                        t_eval: np.ndarray[float],
                                        count_m: int,
                                        count_e: int) -> tuple[float, str]:
    """
    Helper to evaluate the next time for prediction in the filter and whether to provide update.

    Args:
        t_measurements (np.ndarray[float]): vector of measurement times
        t_eval (np.ndarray[float]): vector of evaluation times
        count_m (int): index of value in t_measurements 
        count_e (int): index of value in t_eval

    Returns
        next_time, type (tuple): tuple containing next time and type (whether to provide update). 
    """

    if (count_m < t_measurements.size) and (count_e < t_eval.size) :
        t_meas = t_measurements[count_m]
        t_ev = t_eval[count_e]

        if t_meas == t_ev:
            type = "both"
            next_time = t_meas

        elif t_meas < t_ev:
            type = "measure"
            next_time = t_meas

        elif t_meas > t_ev:
            type = "eval"
            next_time = t_ev
        else:
            raise ValueError("comparison between floats was not equal, <, or >. Something is wrong...")
    elif (count_m < t_measurements.size) and (count_e >= t_eval.size):
        t_meas = t_measurements[count_m]
        type = "measure"
        next_time = t_meas
    elif (count_m >= t_measurements.size) and (count_e < t_eval.size):
        t_ev = t_eval[count_e]
        type = "eval"
        next_time = t_ev
    else:
        raise NotImplementedError

    return next_time, type