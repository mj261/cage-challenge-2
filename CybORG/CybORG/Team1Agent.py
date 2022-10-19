import inspect
from CybORG.Shared import Results
from stable_baselines3 import PPO
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG import CybORG


class Team1Agent:

    def train(self, env):
        self.model = PPO('MlpPolicy', env)
        self.model.learn(total_timesteps=2000)
        self.model.save("Team1")
        return self.model

    def get_action(self, observation, action_space):
        if self.model is None:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
            cyborg = ChallengeWrapper(env=CybORG(path, 'sim'), agent_name='Blue')
            self.model = self.train(cyborg)
        action, _states = self.model.predict(observation)
        return action

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, model_file: str = None):
        if model_file is not None:
            self.model = PPO.load(model_file)
        else:
            self.model = None
