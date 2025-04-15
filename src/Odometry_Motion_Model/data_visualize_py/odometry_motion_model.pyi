class ControlCommand:
    rot_1: float
    rot_2: float
    trans: float
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...

class Noise:
    rotational_noise: float
    translational_noise: float
    def __init__(self, arg0: float, arg1: float) -> None: ...

class OdometryMotionModel:
    def __init__(self, arg0: Noise, arg1: Noise, arg2: Noise, arg3: float) -> None: ...
    def get_posterior_probability(self, arg0: Pose, arg1: Pose, arg2: ControlCommand) -> float: ...
    def sample_motion(self, arg0: Pose, arg1: ControlCommand) -> Pose: ...

class Pose:
    theta: float
    x: float
    y: float
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None: ...
