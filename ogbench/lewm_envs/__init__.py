"""Self-contained LeWM evaluation environments shipped with OGBench.

The implementations were migrated from ``pickxiguapi/stable-worldmodel`` at
commit ``b874c7ef9cc96f099407b7bfb4e20c4c6e0b1f8f``.  They live here so that
training and evaluation use one OGBench checkout and one Python environment.
"""

from gymnasium.envs.registration import register, registry


ENV_IDS = {
    'cube': 'ogbench-lewm/CubeSingle-v0',
    'pusht': 'ogbench-lewm/PushT-v1',
    'tworoom': 'ogbench-lewm/TwoRoom-v1',
    'reacher': 'ogbench-lewm/Reacher-v0',
}


def _register(env_id, entry_point):
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point)


_register(ENV_IDS['cube'], 'ogbench.lewm_envs.cube:CubeEnv')
_register(ENV_IDS['pusht'], 'ogbench.lewm_envs.pusht:PushT')
_register(ENV_IDS['tworoom'], 'ogbench.lewm_envs.two_room:TwoRoomEnv')
_register(
    ENV_IDS['reacher'],
    'ogbench.lewm_envs.dmcontrol.reacher:ReacherDMControlWrapper',
)


def __getattr__(name):
    if name == 'CubeEnv':
        from ogbench.lewm_envs.cube import CubeEnv

        return CubeEnv
    if name == 'PushT':
        from ogbench.lewm_envs.pusht import PushT

        return PushT
    if name == 'TwoRoomEnv':
        from ogbench.lewm_envs.two_room import TwoRoomEnv

        return TwoRoomEnv
    if name == 'ReacherDMControlWrapper':
        from ogbench.lewm_envs.dmcontrol.reacher import ReacherDMControlWrapper

        return ReacherDMControlWrapper
    raise AttributeError(name)


__all__ = ['ENV_IDS', 'CubeEnv', 'PushT', 'ReacherDMControlWrapper', 'TwoRoomEnv']
