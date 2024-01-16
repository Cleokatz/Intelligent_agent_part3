# Projektpraktikum Information Retrieval Agents
# Institut für Informationssysteme der Universitaet zu Luebeck
#	Magnus Bender, 2022

import os

from typing import List

class GoldStandard():

	"""
		Loads a gold standard file in the format used in the project internship.
	"""

	def __init__(self, gold_file:str):
		"""
			Args:
				gold_file (str): Path to the gold standard file to load
		"""
		self.gold_file = gold_file
		if not os.path.isfile(self.gold_file):
			raise FileNotFoundError('Gold standard file not found: ' + gold_file)
		self.keys, self.values = self._load_gold()

	def _load_gold(self):
		keys, values = {}, {}
		counter = -1
		new_key = True

		with open(self.gold_file, 'r', errors='ignore') as f:
			for line in f:
				if len(line) > 0:
					title = line.strip()
					if line[0] != '\t':
						if new_key:
							counter += 1
						keys[title] = counter
						new_key = False
					else:
						if counter not in values:
							values[counter] = []
						if len(title) > 0:
							values[counter].append(title)
						new_key = True

		return keys, values

	def get_corrects(self, query_title:str) -> List[str]:
		"""
			Returns a list of correct titles for a given query title.

			Args:
				query_title (str): The query title to get the correct titles for

			Returns:
				The correct titles or an empty list if the query title is not in the gold standard
		"""
		return self.values[self.keys[query_title]] if query_title in self.keys else []		
