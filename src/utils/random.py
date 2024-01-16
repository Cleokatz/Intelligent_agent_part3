# Projektpraktikum Information Retrieval Agents
# Institut für Informationssysteme der Universitaet zu Luebeck
#	Magnus Bender, 2021

import random, time

class Random():

	_RANDOM_SEED = int(time.time() * 1000)

	def set_seed(seed):
		'''
			Set the global Random seed! (do only once per run and before creating any objects!)
			Else will use scripts init time
		'''
		Random._RANDOM_SEED = seed

	def get_seed():
		'''
			Get the currently used seed
		'''
		return Random._RANDOM_SEED

	def get_generator():
		'''
			Get a seeded fresh Python Random object (always use this in the entire project to get randomness 
			or act based on value of seed, e.g. as random index for initialization)
		'''
		return random.Random(Random._RANDOM_SEED)