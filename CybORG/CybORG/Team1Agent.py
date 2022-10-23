# CSE 8713
# Matthew Jones & Morgan Reese


import inspect
from stable_baselines3 import PPO
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG import CybORG


# Blue Agent for Team 1
class Team1Agent:

    # Function to train the agent
    def train(self, env):
        # Utilize stable_baselines3 PPO MLP Policy
        self.model = PPO('MlpPolicy', env)
        # Train agent over 2000 timesteps
        self.model.learn(total_timesteps=2000)
        # Save Model named "Team1"
        self.model.save("Team1")
        return self.model

    # Function to get scenario and start training
    def get_action(self, observation, action_space):
        # If model does not exist, create one
        if self.model is None:
            # Get filepath
            path = str(inspect.getfile(CybORG))
            # Add scenario file to filepath
            path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
            # Create a blue agent using the CybORG environment
            cyborg = ChallengeWrapper(env=CybORG(path, 'sim'), agent_name='Blue')
            # Start training the agent
            self.model = self.train(cyborg)
        # Observe the agent
        action, _states = self.model.predict(observation)
        return action

    # Function to end the episode
    def end_episode(self):
        pass

    # Set the initial values
    def set_initial_values(self, action_space, observation):
        pass

    # Initial Function
    def __init__(self, model_file: str = None):
        # If model does not exist, create one
        if model_file is not None:
            self.model = PPO.load(model_file)
        else:
            self.model = None
