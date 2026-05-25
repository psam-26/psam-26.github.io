from typing import Any, Callable, List
from src.psam.core.interfaces import IVLAWhiteBox
from src.psam.evaluation.continuous_metrics import ContinuousRoboticMetrics
from abc import ABC, abstractmethod

class EnvironmentStateObserver(ABC):
    @abstractmethod
    def on_step_completed(self, step: int, state: Any, action: Any) -> None:
        pass

class TrajectoryLogger(EnvironmentStateObserver):
    def __init__(self):
        self._action_buffer = []
        self._state_buffer = []

    def on_step_completed(self, step: int, state: Any, action: Any) -> None:
        self._action_buffer.append(action)
        self._state_buffer.append(state)
        
    def retrieve_action_history(self) -> List[Any]:
        return self._action_buffer

class ClosedLoopSimulator:
    def __init__(self, environment_factory: Callable, metrics_engine: ContinuousRoboticMetrics):
        self.env_factory = environment_factory
        self.metrics = metrics_engine
        self.observers: List[EnvironmentStateObserver] = []

    def register_observer(self, observer: EnvironmentStateObserver) -> None:
        self.observers.append(observer)

    def notify_observers(self, step: int, state: Any, action: Any) -> None:
        for observer in self.observers:
            observer.on_step_completed(step, state, action)

    def execute_rollout(self, model: IVLAWhiteBox, patch: Any, task_instruction: str, max_steps: int) -> dict:
        env_instance = self.env_factory()
        current_observation = env_instance.reset()
        
        oracle_buffer = []
        logger = TrajectoryLogger()
        self.register_observer(logger)
        
        for step in range(max_steps):
            corrupted_observation = self._inject_adversarial_patch_to_scene(current_observation, patch)
            hardware_degraded_observation = self._simulate_sensor_noise_and_blur(corrupted_observation)
            
            adversarial_action = model.predict_kinematics(hardware_degraded_observation, task_instruction)
            oracle_action = self._query_oracle_policy(env_instance)
            oracle_buffer.append(oracle_action)
            
            current_observation, is_terminal, transition_info = env_instance.step(adversarial_action)
            self.notify_observers(step, current_observation, adversarial_action)
            
            if is_terminal or self._check_kinematic_safety_bounds(adversarial_action):
                break
                
        return self._compile_rollout_statistics(logger.retrieve_action_history(), oracle_buffer, transition_info)

    def _inject_adversarial_patch_to_scene(self, obs: Any, patch: Any) -> Any:
        pass

    def _simulate_sensor_noise_and_blur(self, obs: Any) -> Any:
        pass

    def _query_oracle_policy(self, env: Any) -> Any:
        pass

    def _compile_rollout_statistics(self, adv_traj: List[Any], gt_traj: List[Any], info: Any) -> dict:
        pass

    def _check_kinematic_safety_bounds(self, action: Any) -> bool:
        pass