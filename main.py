from pathlib import Path

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT,SIMPLE_MOVEMENT,RIGHT_ONLY

from ia import IA
from population import Population
from ramReader import ramReader
from viewer import Viewer, Vector

env = gym_super_mario_bros.make('SuperMarioBros-v0')

env = JoypadSpace(env, RIGHT_ONLY)

viewer = Viewer("Window",Vector(352,240))
ramreader = ramReader(env)
population = Population(IA,16,1, viewer.sensorsN , env.action_space.n)

population.individus[0].load(Path("checkpoints/mario_net_best.pt"))

while not( population.finished() ):
    env.reset()
    ramreader.reset()
    life = 0
    distance = 0

    actual = population.get_actual()
    cooldown = 0
    while True:
        background = ramreader.getBackground()
        entities = ramreader.getEntities()
        player = ramreader.getEntity(0)

        viewer.update(background + entities, player)
        sensors = viewer.getTorchSensors()
        viewer.render()
        env.render()
        action = actual.act(sensors)

        next_state, reward, done, info = env.step(action)

        actual.reward(reward)

        life = max( info['life'], life )

        if info['x_pos'] <= distance:
            cooldown += 1
        else:
            cooldown = 0

        distance = max( info['x_pos'], distance )

        if done or info['flag_get'] or life > info['life'] or cooldown > 300:
            break

    population.step()

viewer.destroy()
env.close()

save_dir = Path('checkpoints')
if not( save_dir.exists() ):
    save_dir.mkdir(parents=True)

population.individus[0].save(save_dir)