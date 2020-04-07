import time

import pybullet

from kukaStackPrimitiveEnv import KukaStackPrimitiveEnv


def planExecution(plan):
	for op in plan:
		ops = op.split(' ')
		if ops[0] == 'pick-up':
			assert len(ops) == 2
			blk = ops[1]
			env._grasp(blk, *graspOffset)
		elif ops[0] == 'stack':
			assert len(ops) == 3
			blk_stack = ops[1]
			blk_support = ops[2]
			preds = env.retrievePredicate()
			assert preds['inHand'][blk_stack]
			env._put(blk_support, *putOffset)
		elif ops[0] == 'unstack':
			assert len(ops) == 3
			blk_stack = ops[1]
			blk_support = ops[2]
			preds = env.retrievePredicate()
			assert preds['on']['{}_{}'.format(blk_support, blk_stack)]
			env._grasp(blk_stack, *graspOffset)
		else:
			raise NotImplementedError


env = KukaStackPrimitiveEnv(renders=True)
pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-30, cameraPitch=-52, cameraTargetPosition=[0, 0, 0])

env.reset()

for _ in range(200):
	env._p.stepSimulation()

graspOffset = [[0, 0, 0], 0.28, 0]  # black magic 
putOffset = [[0, 0, 0], 0.4, 0]

goal = '(:goal (AND (ON white green) (ON yellow red)))'
env.generatePDDLInit('problem.pddl', goal)
plan = env.ffPlan()
planExecution(plan)

goal = '(:goal (AND (ON white yellow) (ON green white)))'
env.generatePDDLInit('problem.pddl', goal)
plan = env.ffPlan()
planExecution(plan)

time.sleep(10)
