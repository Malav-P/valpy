import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, List

import rudolfpy as rd

class ValidationModel:
    """
    Class for validating sensor tasking performance

    Args:
        u (np.ndarray[int]): tasking schedule / control

    """
    def __init__(self,
                 groundtruth_observers: np.ndarray[float],
                 groundtruth_targets: np.ndarray[float],
                 estimate_observers: np.ndarray[float],
                 estimate_targets: np.ndarray[float],
                 P0_observers: np.ndarray[float],
                 P0_targets: np.ndarray[float],
                 filter: rd.BaseFilter , 
                 timestep: float):
        
        # if u.ndim != 3:
        #     raise ValueError("Control tensor must have 3 dimensions, i for observer, j for target, and k for timestep")
        
        if not isinstance(filter, rd.BaseFilter):
            raise TypeError("filter must be an instance rd.BaseFilter")

        self.groundtruth_observers = groundtruth_observers
        self.groundtruth_targets = groundtruth_targets
        self.estimate_observers = estimate_observers
        self.estimate_targets = estimate_targets
        self.P0_observers = P0_observers
        self.P0_targets = P0_targets
        self.filter = filter
        self.timestep = timestep
        
    

    def _get_t_measurements_target(self, u: np.ndarray, target_idx: int) -> tuple[np.ndarray[float], np.ndarray[int]]:
        """
        Get a list of measurement times for the specified target, according to the control

        Args:
            u (np.ndarray): 3d control array
            target_idx (int): index of the target to get measurements for

        Returns:
            t_measurements (np.ndarray[float]): a vector of measurement times for the target
            observer_indices (np.ndarray[int]): a vector of observer indices at the relevant times for the target
        """
        u_j = u[:, target_idx, :]

        indices = np.where(u_j == 1)


        t_measurements = self.timestep * (np.sort(indices[1]) + 0.5)
        observer_indices = indices[0]

        return t_measurements, observer_indices
    
    def _get_measurement_info_target(self,
                                     u: np.ndarray,
                                     target_idx: int,
                                     filter_observers: Optional[bool] = False,
                                     meas_model_sigma: Optional[float] = None,
                                     dt_exposure: Optional[float] = None) -> tuple[np.ndarray, list]:
        """
        Get a measurement times and list of parameters for target measurement. Use default values.

        Args:
            u (np.ndarray): 3d control tensor
            target_idx (int): index of chosen target        
            filter_observers (bool): Whether to filter observers. Defaults to False.
            meas_model_sigma (float): standard deviation of measurement model noise. If None, use default values.
            dt_exposure (float): time of exposure for sensor measurement. If None, use default values.
        Returns:
            t_measurements (np.ndarray[float]): vector of measurement times
            params_measurements (list): list of measurement parameters for specified target
        """

        t_measurements, observer_indices = self._get_t_measurements_target(u=u, target_idx=target_idx)

        if filter_observers:
            # TODO
            raise NotImplementedError
        
        observer_histories = {}
        for unique_observer_idx in np.unique(observer_indices):
            observer_histories[unique_observer_idx] = self._get_observer_history_groundtruth(u=u, observer_idx=unique_observer_idx, t_eval=t_measurements)

        x_observers = []

        match self.filter.measurement_model.name:
            case "Angle":
                sigma = np.deg2rad(0.5) if meas_model_sigma is None else meas_model_sigma # std of 0.5 degrees in angle measurement
                keep = 3

                for i in range(len(t_measurements)):
                    observer_idx = observer_indices[i]
                    observer_state, _ = observer_histories[observer_idx][i]
                    x_observers.append(observer_state[:keep])

                params_measurements = [[x_observer, sigma] for x_observer in x_observers ]
                
            case "Angle_AngleRate":
                sigma = np.deg2rad(0.5) if meas_model_sigma is None else meas_model_sigma # std of 0.5 degrees in angle measurement
                dt = 3600 / self.filter.dynamics.TU if dt_exposure is None else dt_exposure # one hour exposure time for measurement
                keep = 6


                for i in range(len(t_measurements)):
                    observer_idx = observer_indices[i]
                    observer_state, _ = observer_histories[observer_idx][i]
                    x_observers.append(observer_state[:keep])

                params_measurements = [[x_observer, sigma, dt] for x_observer in x_observers ]

            case "Optical":
                sigma = 1 if meas_model_sigma is None else meas_model_sigma # 1 pixel std dev in position measurement 
                dt = 3600 / self.filter.dynamics.TU if dt_exposure is None else dt_exposure# one hour exposure time for measurement
                keep = 6

                for i in range(len(t_measurements)):
                    observer_idx = observer_indices[i]
                    observer_state, _ = observer_histories[observer_idx][i]
                    x_observers.append(observer_state[:keep])
                
                params_measurements = [[x_observer, sigma, dt] for x_observer in x_observers ]


            case "PositionVector":
                # TODO
                raise NotImplementedError
            
        return t_measurements, params_measurements


    
    def _get_t_measurements_observer(self,
                                     u: np.ndarray,
                                     observer_idx: int,
                                     num_measurements: Optional[int] = None) -> np.ndarray[float]:
        """
        Get a list of measurement times for the specified observer. Assume that the requested number of measurements is 
        equally distributed over the time span specified by the control.

        Args:
            u (np.ndarray): 3d control array
            observer_idx (int): index of the observers to get measurements for
            num_measurements (int): number of measurements requested over simulation time. Must be > 0.

        Returns:
            t_measurements (np.ndarray[float64]): a vector of measurement times for the observer
        """
        num_steps = u.shape[2]

        if num_measurements is None:
            num_measurements = num_steps

        elif num_measurements < 1:
            raise ValueError("Argument `num_measurements` must be greater than 0.")
        

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
                                          u: np.ndarray,
                                          observer_idx:int,
                                          t_eval: Optional[np.ndarray] = None) -> list[tuple]:
        """
        Get observer state history, using the ground truth observer initial condition

        Args:
            u (np.ndarray): control tensor 3d
            observer_idx (int): index of the observer to propagate
            t_eval (np.ndarray[float]): vector of times to evaluate solution at

        Returns:
            observer_history (list[tuple]): observer history as a list of (x, P) tuples.
        """
        if t_eval is None:
            num_steps = u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        if type(t_eval) is not np.ndarray:
            raise TypeError("`t_eval` must be of type np.ndarray")

        x0 = self.groundtruth_observers[observer_idx]
        dim_x = x0.size


        sol_true = self.filter.dynamics.solve([0.0, t_eval[-1]], x0, t_eval=t_eval)

        observer_history = [(sol_true.y[:dim_x, idx], None) for idx in range(t_eval.size)]

        return observer_history

    def _get_target_history_groundtruth(self,
                                        u: np.ndarray,
                                        target_idx:int,
                                        t_eval: Optional[np.ndarray] = None) -> list[tuple]:
        """
        Get target state history, using the ground truth target initial condition

        Args:
            u (np.ndarray): control tensor 3d
            target_idx (int): index of the target to propagate
            t_eval (np.ndarray[float]): vector of times to evaluate solution at

        Returns:
            sol_true (OdeResult): returned object from `scipy.integrate.solve_ivp`
        """
        if t_eval is None:
            num_steps = u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        if type(t_eval) is not np.ndarray:
            raise TypeError("`t_eval` must be of type np.ndarray")

        x0 = self.groundtruth_targets[target_idx]
        dim_x = x0.size

        sol_true = self.filter.dynamics.solve([0.0, t_eval[-1]], x0, t_eval=t_eval)

        target_history = [(sol_true.y[:dim_x, idx], None) for idx in range(t_eval.size)]

        return target_history
    
    def _get_observer_history_filter(self,
                                     u: (np.ndarray),
                                     observer_idx: int,
                                     params_measurements: list,
                                     t_measurements: np.ndarray[float],
                                     t_eval: Optional[np.ndarray[float]] = None) -> list[tuple]:
        """
        Get the observer state history, using a filter to estimate the observer position

        Args:
            u (np.ndarray): 3d control tensor
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
            num_steps = u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        observer_history_gt = self._get_observer_history_groundtruth(observer_idx=observer_idx, t_eval=t_measurements)
        observer_history = []

        x0 = self.estimate_observers[observer_idx]
        P0 = self.P0_observers[observer_idx]

        # initialize filter
        self.filter.initialize(t = 0, x0 = x0, P0 = P0)

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
                    y, R = self.filter.measurement_model.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m])
                    count_m += 1
                case "both":
                    x_true = observer_history_gt[count_m][0]
                    y, R = self.filter.measurement_model.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m])
                    x = self.filter.x
                    P = self.filter.P
                    observer_history.append((deepcopy(x), deepcopy(P)))
                    count_m += 1
                    count_e += 1

        return observer_history
    
    def _get_target_history_filter(self,
                                   u: np.ndarray,
                                   target_idx: int,
                                   params_measurements: list,
                                   t_measurements: Optional[np.ndarray[float]] = None,
                                   t_eval: Optional[np.ndarray[float]] = None,
                                   filter_observers: Optional[bool] = False,
                                   **kwargs) -> list[tuple]:
        """
        Get the target state history at the requested times with the requested measurements

        Args:
            u (np.ndarray): 3d control tensor
            target_idx (int): index of the target to propagate
            params_measurements (list): list of measurement parameters to pass to `func_simulate_measurements` for each measurement taken.
            t_measurements (np.ndarray[float]): vector of times when measurements of target state are taken. If None, defaults to times by control
            t_eval (np.ndarray[float]): vector of times to query the filtered state at. If None, equally distributed amongst time horizon specified in control
            filter_observers (bool): Whether to filter observers. Defaults to False.
            **meas_model_kwargs: keyword arguments to pass to measurement model when getting measurement info for targets

        Returns:
            target_history (list[tuple[np.ndarray, np.ndarray]]): state history of target.

        Notes:
            - if t_measurements is given, params_measurements must have same length as t_measurements. Otherwise, params_measurements is ignored.
        """

        if t_measurements is None:
            t_measurements, params_measurements = self._get_measurement_info_target(u=u,target_idx=target_idx, filter_observers=filter_observers, **kwargs)


        if len(params_measurements) != t_measurements.size:
            raise AssertionError(f"params_measurements (size: {len(params_measurements)}) must be of same size as t_measurements (size: f{t_measurements.size}).")
        

        if t_eval is None:
            num_steps = u.shape[2]
            t_eval = np.linspace(0, self.timestep * num_steps, 300)

        target_history_gt = self._get_target_history_groundtruth(u=u,target_idx=target_idx, t_eval=t_measurements)
        target_history = []

        x0 = self.estimate_targets[target_idx]
        P0 = self.P0_targets[target_idx]

        # initialize filter
        self.filter.initialize(t = 0, x0 = x0, P0 = P0)

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
                    y, R = self.filter.measurement_model.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m])
                    count_m += 1
                case "both":
                    x_true = target_history_gt[count_m][0]
                    y, R = self.filter.measurement_model.func_simulate_measurements(self.filter.t, x_true , params_measurements[count_m])
                    self.filter.update(y, R, params = params_measurements[count_m])
                    x = self.filter.x
                    P = self.filter.P
                    target_history.append((deepcopy(x), deepcopy(P)))
                    count_m += 1
                    count_e += 1


        return target_history
    
    def run(self,
            u: np.ndarray,
            filter_observers: Optional[bool] = False,
            **kwargs) -> Tuple[List[Tuple[np.ndarray[float], np.ndarray[float]]],
                         List[Tuple[np.ndarray[float], np.ndarray[float]]]]: 
        """
        Run the validation model, filtering targets (and observers if necessary). Record history of target and observer states

        Args:
            u (np.ndarray): 3d control tensor
            filter_observers (bool): whether or not to use filtering for observer states
            **meas_model_kwargs: keyword arguments to pass to measurement model when getting measurement info for targets

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
            target_history = self._get_target_history_filter(u=u,
                                                            target_idx=i, 
                                                            params_measurements=None, # will be ignored since t_measurements is None
                                                            t_measurements=None,  # default to measurements provided by control
                                                            t_eval=None, # default to 300 equally spaced points in the time horizon
                                                            filter_observers=filter_observers,
                                                            **kwargs)
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
                observer_history = self._get_observer_history_groundtruth(u=u,
                                                                          observer_idx=j,
                                                                          t_eval=None)
                
                observer_histories.append(observer_history)
                

        
        return target_histories, observer_histories
    

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