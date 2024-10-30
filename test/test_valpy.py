import numpy as np
import pytest
from valpy.model import ValidationModel
import rudolfpy as rd

# to run these tests:
# 1. make sure pytest (conda install pytest) is installed in your virtual environment.
# 2. run `pytest test` from the root dir

## ground truth initial conditions for targets and observers
groundtruth_targets = np.array([[
        1.1540242813087864,
        0.0,
        -0.1384196144071876,
        4.06530060663289e-15,
        -0.21493019200956867,
        8.48098638414804e-15
    ]])
groundtruth_observers = np.array([[
        0.8027692908754149,
        0.0,
        0.0,
        -1.1309830924549648e-14,
        0.33765564334938736,
        0.0
    ]])

# CR3BP dynamics parameters
mu = 1.215058560962404e-02
LU = 384400
TU = 3.751902619517228e+05
VU = LU / TU

# positional uncertainty (is used to compute initial covariance of target and observer state)
sigma_r0 = 100 / LU
sigma_v0 = 0.001 / VU

# measurement model uncertainty in angle measurement
sigma_phi = np.deg2rad(0.1)
# measurement model time of exposure for sensor measurement
dt_meas = 3600/TU

# estimates for (x0, P0) for observer and targets
estimate_observers = np.array([groundtruth_observers[0] + np.array([sigma_r0]*3 + [sigma_v0]*3) * np.random.normal(0, 1, 6)])
estimate_targets = np.array([groundtruth_targets[0] + np.array([sigma_r0]*3 + [sigma_v0]*3) * np.random.normal(0, 1, 6)])
P0_observers = np.array([np.diag([sigma_r0]*3 + [sigma_v0]*3)**2])
P0_targets = np.array([np.diag([sigma_r0]*3 + [sigma_v0]*3)**2])

# create dynamics model
dynamics = rd.DynamicsCR3BP(mu = mu, LU=LU, TU=TU, method='DOP853', rtol = 1e-12, atol = 1e-12)
# create measurement model
meas_model = rd.MeasurementAngleAngleRate()
# process noise
params_Q = [1e-5,]
# create EKF object
filter = rd.ExtendedKalmanFilter(dynamics, meas_model,
                                func_process_noise = rd.unbiased_random_process_3dof,
                                params_Q = params_Q,)
# timestep of simulation
timestep = 0.015

# simulate randome control
u = np.random.randint(low=0, high=2, size=(1, 1, 215))

def test_validation_model_init():

    # create ValidationModel Object
    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    
def test_invalid_u():

    u = np.random.randint(low=0, high=2, size=(1, 215)) # u only has 2 dims instead of 3 which is wrong

    with pytest.raises(ValueError) as excinfo:
        # create ValidationModel Object
        vm = ValidationModel(u=u,
                            groundtruth_observers=groundtruth_observers,
                            groundtruth_targets=groundtruth_targets,
                            estimate_observers=estimate_observers,
                            estimate_targets=estimate_targets,
                            P0_observers=P0_observers,
                            P0_targets=P0_targets,
                            filter=filter,
                            timestep=timestep)
        
    assert excinfo.type is ValueError

def test_invalid_filter():

    filter = 0.5 # not a rd.ExtendedKalmanFilter object

    with pytest.raises(TypeError) as excinfo:
        # create ValidationModel Object
        vm = ValidationModel(u=u,
                            groundtruth_observers=groundtruth_observers,
                            groundtruth_targets=groundtruth_targets,
                            estimate_observers=estimate_observers,
                            estimate_targets=estimate_targets,
                            P0_observers=P0_observers,
                            P0_targets=P0_targets,
                            filter=filter,
                            timestep=timestep)
        
    assert excinfo.type is TypeError

def test_get_t_measurements_target():

    u = np.zeros(shape=(1, 1, 30))

    measure_idxs = [5, 8, 18, 22]

    u[0, 0, measure_idxs] = 1

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    
    tmeasure, obs_indices = vm._get_t_measurements_target(target_idx=0)

    assert tmeasure.size == 4

    for i, idx in enumerate(measure_idxs):
        assert tmeasure[i] == vm.timestep * (idx + 0.5)

    assert not obs_indices.any()

def test_get_measurement_info_target():

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    
    with pytest.raises(NotImplementedError) as excinfo:
        # create ValidationModel Object
        vm._get_measurement_info_target(target_idx=0,
                                        filter_observers=True) # filtering observers is not supported yet
        
    assert excinfo.type is NotImplementedError

    vm._get_measurement_info_target(target_idx=0,
                                    filter_observers=False) # using ground truth observer history should work fine
    
def test_t_measurements_observer():

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    

    num_measure = 1
    tmeasure = vm._get_t_measurements_observer(observer_idx=0,
                                               num_measurements=num_measure)
    assert tmeasure.size == num_measure


    num_measure = 0
    with pytest.raises(ValueError) as excinfo:
        # create ValidationModel Object
        tmeasure = vm._get_t_measurements_observer(observer_idx=0,
                                                   num_measurements=num_measure)
    assert excinfo.type is ValueError


def test_get_measurement_info_observer():

    # create measurement model
    meas_model = rd.MeasurementPosition()
    # create EKF object
    filter_ = rd.ExtendedKalmanFilter(dynamics, meas_model,
                                    func_process_noise = rd.unbiased_random_process_3dof,
                                    params_Q = params_Q,)

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter_,
                        timestep=timestep)
    

    with pytest.raises(NotImplementedError) as excinfo:
        vm._get_measurement_info_observer(observer_idx=0)

    assert excinfo.type is NotImplementedError

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    
    tmeasure, param = vm._get_measurement_info_observer(observer_idx=0)

    assert tmeasure.size == vm.u.shape[2]

def test_get_observer_history_groundtruth():

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    
    observer_history = vm._get_observer_history_groundtruth(observer_idx=0,
                                                            t_eval=None)
    assert len(observer_history) == 300
    assert all([obj[1] is None for obj in observer_history]) # ground truth has no covariance history

    observer_history = vm._get_observer_history_groundtruth(observer_idx=0,
                                                            t_eval=np.linspace(0, 0.1, 200))
    assert len(observer_history) == 200
    assert all([obj[1] is None for obj in observer_history]) # ground truth has no covariance history

    with pytest.raises(TypeError) as excinfo:
        vm._get_observer_history_groundtruth(observer_idx=0,
                                             t_eval = [0.0, 1.0, 3.0])
    
    assert excinfo.type is TypeError

def test_get_target_history_groundtruth():

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    
    target_history = vm._get_target_history_groundtruth(target_idx=0,
                                                            t_eval=None)
    assert len(target_history) == 300
    assert all([obj[1] is None for obj in target_history]) # ground truth has no covariance history

    target_history = vm._get_target_history_groundtruth(target_idx=0,
                                                            t_eval=np.linspace(0, 0.1, 200))
    assert len(target_history) == 200
    assert all([obj[1] is None for obj in target_history]) # ground truth has no covariance history

    with pytest.raises(TypeError) as excinfo:
        vm._get_target_history_groundtruth(target_idx=0,
                                             t_eval = [0.0, 1.0, 3.0])
    
    assert excinfo.type is TypeError

def test_run():

    vm = ValidationModel(u=u,
                        groundtruth_observers=groundtruth_observers,
                        groundtruth_targets=groundtruth_targets,
                        estimate_observers=estimate_observers,
                        estimate_targets=estimate_targets,
                        P0_observers=P0_observers,
                        P0_targets=P0_targets,
                        filter=filter,
                        timestep=timestep)
    
    thist, obshist = vm.run(filter_observers=False)

    assert all(item[1].shape == (6, 6) and item[0].shape == (6,) for item in thist)
    assert all(item[1] is None and item[0].shape == (6,) for item in obshist)

    assert len(thist) == 300
    assert len(obshist) == 300
    