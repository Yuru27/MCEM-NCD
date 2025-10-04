from functools import partial
from .multiagentenv import MultiAgentEnv
from .particle import Particle



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["particle"] = partial(env_fn, env=Particle)