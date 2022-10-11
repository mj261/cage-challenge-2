import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Agents import *
from CybORG.Shared.Actions import *

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

red_agent = RedMeanderAgent
env = CybORG(path, 'sim', agents={'Red': red_agent})

agent = BlueReactRemoveAgent()

results = env.reset('Blue')
obs = results.observation
action_space = results.action_space

for i in range(12):
    action = agent.get_action(obs, action_space)
    results = env.step(action=action, agent='Blue')
    obs = results.observation
    print(env.get_last_action('Blue'))

