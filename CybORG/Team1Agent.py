import inspect
from CybORG.Shared import Results
from stable_baselines3 import PPO
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG import CybORG


class Team1Agent:

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        if self.model is None:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
            cyborg = ChallengeWrapper(env=CybORG(path, 'sim'), agent_name='Blue')
            self.model = PPO('MlpPolicy', cyborg)
            self.model.learn(total_timesteps=2000)
            self.model.save("Team1")
        action, _states = self.model.predict(observation)
        return action

    def end_episode(self):
        """Allows an agent to update its internal state"""
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, model_file: str = None):
        if model_file is not None:
            self.model = PPO.load(model_file)
        else:
            self.model = None
