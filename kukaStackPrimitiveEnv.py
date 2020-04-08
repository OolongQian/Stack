import math
import os

import gym
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_envs.bullet.kuka as kuka
from gym.utils import seeding


class KukaStackPrimitiveEnv(gym.Env):
	"""
	This robot manipulation environment using primitive is different from the one using reinforcement learning.
	The self._kuka.applyAction() function is not appropriate, because its argument is the relative displacement.
	"""
	cubeColors = ['green', 'red', 'white', 'yellow']
	
	def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=1, isEnableSelfCollision=True, renders=False,
	             isDiscrete=False, maxSteps=100):
		
		self._isDiscrete = isDiscrete
		self._timeStep = 1. / 240.
		self._urdfRoot = urdfRoot
		self._actionRepeat = actionRepeat
		self._isEnableSelfCollision = isEnableSelfCollision
		self._observation = []
		self._envStepCounter = 0
		self._renders = renders
		self._maxSteps = maxSteps
		self.terminated = 0
		self._cam_dist = 1.3
		self._cam_yaw = 180
		self._cam_pitch = -40
		
		action_high = np.array([0.01, 0.01, 0.01, 0.01, math.pi / 2,  # grasp: x, y, z, h, ang
		                        0.01, 0.01, 0.01, 0.01, math.pi / 2])  # put: x, y, z, h, ang
		self.action_space = gym.spaces.Box(high=action_high, low=-action_high)
		
		observation_high = np.array([1.])
		self.observation_space = gym.spaces.Box(high=observation_high, low=-observation_high)
		
		self._p = p
		
		if self._renders:
			p.connect(p.GUI)
			p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
		else:
			p.connect(p.DIRECT)
		
		self.seed()
		self.reset()
	
	def reset(self):
		self.terminated = 0
		p.resetSimulation()
		p.setPhysicsEngineParameter(numSolverIterations=150)
		p.setTimeStep(self._timeStep)
		print(self._urdfRoot)
		p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
		self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
		                           0.000000, 0.000000, 0.0, 1.0)
		
		self.blockUids = {}
		for i, color in enumerate(self.cubeColors):
			xpos = 0.6 + 0.05 * (i - 2)  # to the left of the table
			ypos = 0.1 + 0.1 * (i - 2)  # closer to the camera
			ang = 3.14 * 0.5 + 3.1415925438 * 0
			orn = p.getQuaternionFromEuler([0, 0, ang])
			self.blockUids[color] = p.loadURDF('./assets/cube_{}.urdf'.format(color), xpos, ypos, -0.15, orn[0], orn[1],
			                                   orn[2], orn[3])
		p.setGravity(0, 0, -10)
		self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
		# p.removeBody(self._kuka.trayUid)
		self._envStepCounter = 0
		p.stepSimulation()
		
		# self._observation = self.getExtendedObservation()
		# return np.array(self._observation)
		observation = np.array([1.])
		return observation
	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def getExtendedObservation(self):
		self._observation = self._kuka.getObservation()  # only get observation from kuka.
		return self._observation
	
	@staticmethod
	def _normalizeEndEffector(endEffectorPos, endEffectorAngle, fingerAngle):
		if endEffectorPos[0] > 0.65:
			endEffectorPos[0] = 0.65
		elif endEffectorPos[0] < 0.50:
			endEffectorPos[0] = 0.50
		if endEffectorPos[1] < -0.17:
			endEffectorPos[1] = -0.17
		elif endEffectorPos[1] > 0.22:
			endEffectorPos[1] = 0.22
		
		highAngle = 1.5
		if endEffectorAngle > highAngle:
			endEffectorAngle = highAngle
		elif endEffectorAngle < -highAngle:
			endEffectorAngle = -highAngle
		
		if fingerAngle > highAngle:
			fingerAngle = highAngle
		elif fingerAngle < -highAngle:
			fingerAngle = -highAngle
		
		return endEffectorPos, endEffectorAngle, fingerAngle
	
	def _handUpOffset(self, endEffectorPos, height):
		endEffectorPos[2] += height
		return endEffectorPos
	
	def step(self, action):
		self._grasp('white', dPos=action[:3], dh=action[3], ang=action[4])
		self._put('red', dPos=action[5:8], dh=action[8], ang=action[9])
		
		observation = np.array([1.])
		reward = 0
		if self._on('white', 'red'):
			reward += 1000
		done = True
		
		return observation, reward, done, {}
	
	def _grasp(self, blockColor, dPos, dh, ang):
		pos, blkOrn = self._p.getBasePositionAndOrientation(self.blockUids[blockColor])
		endEffectorPos = list(pos)
		endEffectorAngle = 0
		fingerAngle = 0
		
		endEffectorPos, endEffectorAngle, fingerAngle = self._normalizeEndEffector(endEffectorPos, endEffectorAngle,
		                                                                           fingerAngle)
		
		endEffectorPos[0] += dPos[0]
		endEffectorPos[1] += dPos[1]
		endEffectorPos[2] += dPos[2]
		
		graspHeightOffset = dh
		endEffectorPos = self._handUpOffset(endEffectorPos, graspHeightOffset)  # empirical value, height above block
		
		pos = endEffectorPos
		orn = p.getQuaternionFromEuler([0, -math.pi, 0])
		
		jointPoses = p.calculateInverseKinematics(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex, pos, orn,
		                                          jointDamping=self._kuka.jd)
		
		for i in range(self._kuka.kukaEndEffectorIndex + 1):
			p.setJointMotorControl2(bodyUniqueId=self._kuka.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
			                        targetPosition=jointPoses[i], targetVelocity=0, force=self._kuka.maxForce,
			                        maxVelocity=self._kuka.maxVelocity, positionGain=0.3, velocityGain=1)
		
		# empirical duration for position control execution
		for _ in range(500):
			self._p.stepSimulation()
		
		# rotate wrist and close fingers
		handAngle = p.getEulerFromQuaternion(blkOrn)[2] + ang
		p.setJointMotorControl2(self._kuka.kukaUid, 7, p.POSITION_CONTROL, targetPosition=handAngle, force=5)
		for _ in range(100):
			self._p.stepSimulation()
		
		fingerAngle = 0
		tipAngle = 0
		
		p.setJointMotorControl2(self._kuka.kukaUid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle,
		                        force=self._kuka.fingerAForce)
		p.setJointMotorControl2(self._kuka.kukaUid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle,
		                        force=self._kuka.fingerBForce)
		
		p.setJointMotorControl2(self._kuka.kukaUid, 10, p.POSITION_CONTROL, targetPosition=tipAngle,
		                        force=self._kuka.fingerTipForce)
		p.setJointMotorControl2(self._kuka.kukaUid, 13, p.POSITION_CONTROL, targetPosition=tipAngle,
		                        force=self._kuka.fingerTipForce)
		
		for _ in range(100):
			self._p.stepSimulation()
		
		# getup!
		endEffectorPos[2] += 0.2
		orn = orn
		jointPoses = p.calculateInverseKinematics(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex, pos, orn,
		                                          jointDamping=self._kuka.jd)
		
		for i in range(self._kuka.kukaEndEffectorIndex + 1):
			p.setJointMotorControl2(bodyUniqueId=self._kuka.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
			                        targetPosition=jointPoses[i], targetVelocity=0, force=self._kuka.maxForce,
			                        maxVelocity=self._kuka.maxVelocity, positionGain=0.3, velocityGain=1)
		
		# empirical duration for position control execution
		for _ in range(200):
			self._p.stepSimulation()
	
	def _put(self, blockColor, dPos, dh, ang):
		pos, blkOrn = self._p.getBasePositionAndOrientation(self.blockUids[blockColor])
		endEffectorPos = list(pos)
		endEffectorAngle = 0
		fingerAngle = 0
		
		endEffectorPos, endEffectorAngle, fingerAngle = self._normalizeEndEffector(endEffectorPos, endEffectorAngle,
		                                                                           fingerAngle)
		
		endEffectorPos[0] += dPos[0]
		endEffectorPos[1] += dPos[1]
		endEffectorPos[2] += dPos[2]
		
		putHeightOffset = dh
		endEffectorPos = self._handUpOffset(endEffectorPos, putHeightOffset)  # empirical value, height above block
		
		pos = endEffectorPos
		orn = p.getQuaternionFromEuler([0, -math.pi, 0])
		
		pos[1] += 0.03  # feedback compensation, black magic.
		
		# jointPoses = p.calculateInverseKinematics(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex, pos, orn,
		#                                           jointDamping=self._kuka.jd)
		jointPoses = p.calculateInverseKinematics(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex, pos, orn,
		                                          jointDamping=self._kuka.jd)
		
		for i in range(self._kuka.kukaEndEffectorIndex + 1):
			p.setJointMotorControl2(bodyUniqueId=self._kuka.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
			                        targetPosition=jointPoses[i], targetVelocity=0, force=self._kuka.maxForce,
			                        maxVelocity=self._kuka.maxVelocity, positionGain=0.3, velocityGain=1)
		
		# empirical duration for position control execution
		for _ in range(500):
			self._p.stepSimulation()
		
		# rotate wrist and  open fingers
		handAngle = p.getEulerFromQuaternion(blkOrn)[2] + ang
		p.setJointMotorControl2(self._kuka.kukaUid, 7, p.POSITION_CONTROL, targetPosition=handAngle,
		                        force=self._kuka.maxForce)
		for _ in range(100):
			self._p.stepSimulation()
		
		fingerAngle = 0.3
		tipAngle = 0
		p.setJointMotorControl2(self._kuka.kukaUid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle,
		                        force=self._kuka.fingerAForce)
		p.setJointMotorControl2(self._kuka.kukaUid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle,
		                        force=self._kuka.fingerBForce)
		
		p.setJointMotorControl2(self._kuka.kukaUid, 10, p.POSITION_CONTROL, targetPosition=tipAngle,
		                        force=self._kuka.fingerTipForce)
		p.setJointMotorControl2(self._kuka.kukaUid, 13, p.POSITION_CONTROL, targetPosition=tipAngle,
		                        force=self._kuka.fingerTipForce)
		
		for _ in range(100):
			self._p.stepSimulation()
		
		# reset hand up
		putHeightOffset = 0.2
		endEffectorPos = self._handUpOffset(endEffectorPos, putHeightOffset)  # empirical value, height above block
		
		pos = endEffectorPos
		orn = p.getQuaternionFromEuler([0, -math.pi, 0])
		
		jointPoses = p.calculateInverseKinematics(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex, pos, orn,
		                                          jointDamping=self._kuka.jd)
		
		for i in range(self._kuka.kukaEndEffectorIndex + 1):
			p.setJointMotorControl2(bodyUniqueId=self._kuka.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
			                        targetPosition=jointPoses[i], targetVelocity=0, force=self._kuka.maxForce,
			                        maxVelocity=self._kuka.maxVelocity, positionGain=0.3, velocityGain=1)
		
		# empirical duration for position control execution
		for _ in range(500):
			self._p.stepSimulation()
	
	def _putOnTable(self):
		pass
	
	def render(self):
		pass
	
	def getJointInfo(self):
		numJoints = self._p.getNumJoints(self._kuka.kukaUid)
		for i in range(numJoints):
			jointInfo = self._p.getJointInfo(self._kuka.kukaUid, i)
			jointName, jointType = jointInfo[1], jointInfo[2]
			print("joint {}: name {}, type {}.".format(i, jointName, jointType))
	
	def _termination(self):
		raise NotImplementedError
	
	def _reward(self):
		raise NotImplementedError
	
	def retrievePredicate(self):
		predicates = {'on': {}, 'onTable': {}, 'inHand': {}, 'clear': {}}
		
		for blk1 in self.cubeColors:
			for blk2 in self.cubeColors:
				predicates['on']['{}_{}'.format(blk1, blk2)] = self._on(blk1, blk2)
		
		for blk in self.cubeColors:
			predicates['onTable'][blk] = self._onTable(blk)
			predicates['inHand'][blk] = self._inHand(blk)
		
		predicates['handEmpty'] = self._handEmpty()
		
		# derive predicate clear
		for blk in self.cubeColors:
			clear = True
			for other_blk in self.cubeColors:
				if predicates['on']['{}_{}'.format(blk, other_blk)]:
					clear = False
			predicates['clear'][blk] = clear
		
		return predicates
	
	def _on(self, blk1, blk2):
		"""
		blk1, blk2 is color string in self.cubeColors.
		return true if blk2 is on blk1.
		"""
		blk1_id = self.blockUids[blk1]
		blk2_id = self.blockUids[blk2]
		
		# contact
		contactPoints = p.getContactPoints(blk1_id, blk2_id)
		
		# relative position
		blk1_pos, _ = self._p.getBasePositionAndOrientation(blk1_id)
		blk2_pos, _ = self._p.getBasePositionAndOrientation(blk2_id)
		above = blk2_pos[2] > blk1_pos[2]
		
		# stay
		blk1_v = self._p.getBaseVelocity(blk1_id)[0]  # get linear velocity, throw away angular velocity.
		blk2_v = self._p.getBaseVelocity(blk2_id)[0]
		blk1_vlen = math.sqrt(blk1_v[0] ** 2 + blk1_v[1] ** 2 + blk1_v[2] ** 2)
		blk2_vlen = math.sqrt(blk2_v[0] ** 2 + blk2_v[1] ** 2 + blk2_v[2] ** 2)
		stay = blk1_vlen < 1e-1 and blk2_vlen < 1e-1
		
		return len(contactPoints) > 0 and above and stay
	
	def _onTable(self, blk):
		blk_id = self.blockUids[blk]
		
		# contact
		contactPoints = self._p.getContactPoints(blk_id, self.tableUid)
		
		# relative position
		table_pos, _ = self._p.getBasePositionAndOrientation(self.tableUid)
		blk_pos, _ = self._p.getBasePositionAndOrientation(blk_id)
		above = blk_pos[2] > table_pos[2]
		
		# stay
		blk_v, _ = self._p.getBaseVelocity(blk_id)
		blk_vlen = math.sqrt(blk_v[0] ** 2 + blk_v[1] ** 2 + blk_v[2] ** 2)
		stay = blk_vlen < 1e-1
		
		return len(contactPoints) > 0 and above and stay
	
	def _inHand(self, blk):
		blk_id = self.blockUids[blk]
		
		contactPoints = []
		for fid in range(self._p.getNumJoints(self._kuka.kukaUid)):
			contactPoints += self._p.getContactPoints(self._kuka.kukaUid, blk_id, fid)
		
		return len(contactPoints) > 0
	
	def _handEmpty(self):
		# force
		jointReactionForces = self._p.getJointStates(self._kuka.kukaUid, [8, 11])
		
		jPos1 = jointReactionForces[0][0]
		jPos2 = jointReactionForces[1][0]
		
		# almost relaxed
		handEmpty = (math.fabs(jPos1 - (-0.3)) + math.fabs(jPos2 - 0.3)) < 0.05
		return handEmpty
	
	def generatePDDLInit(self, problemPath, goal):
		with open(os.path.join(os.curdir, problemPath), 'w') as f:
			f.writelines(
				['(define (problem BLOCKS-4-0)\n', '(:domain BLOCKS)\n', '(:objects green white red yellow - block)\n'])
			
			f.write('(:INIT')
			predicate = self.retrievePredicate()
			for pred in ['clear', 'onTable']:
				for blk in self.cubeColors:
					if predicate[pred][blk]:
						if pred == 'clear':
							f.write(' (CLEAR {})'.format(blk))
						if pred == 'onTable':
							f.write(' (ONTABLE {})'.format(blk))
			
			# note ON (A, B) in kuka.predicate differs from domain.pddl.
			for blk1 in self.cubeColors:
				for blk2 in self.cubeColors:
					if predicate['on']['{}_{}'.format(blk1, blk2)]:
						f.write(' (ON {} {})'.format(blk2, blk1))
			
			if predicate['handEmpty']:
				f.write(' (HANDEMPTY))\n')
				
			f.write('{}\n'.format(goal))
			f.write(')\n')
			f.flush()
	
	def ffPlan(self):
		"""
		Require properly set path.
		Plan file is named 'sas_plan.1' by default.
		"""
		os.system('./fast-downward-19.12/fast-downward.py --alias seq-sat-lama-2011 ./pddl/domain.pddl ./problem.pddl')
		plan = []
		with open('sas_plan.1', 'r') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip('\n')
				line = line.strip(')')
				line = line.strip('(')
				if line.startswith(';'):
					continue
				plan.append(line)
		# os.system('rm sas_plan.1')
		return plan
