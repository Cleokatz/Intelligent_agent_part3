# Projektpraktikum Information Retrieval Agents
# Institut für Informationssysteme der Universitaet zu Luebeck
#	Magnus Bender, 2021

import os, requests

from src.utils.functions import check_and_create_folder, write_json_file, read_json_file

class Wikipedia():
	"""
		Downloads Wikipedia articles.
		Uses an in-memory and a json cache
	"""

	_CACHE_PATH = os.path.join(
			os.path.dirname(os.path.realpath(__file__)),
			'cache'
		)

	def __init__(self, file_name):
		check_and_create_folder(Wikipedia._CACHE_PATH)
		self.cachefile = os.path.join(Wikipedia._CACHE_PATH, file_name + '.json')

		if os.path.isfile(self.cachefile):
			self.cache = read_json_file(self.cachefile)
		else:
			self.cache = {}

	"""
		Download an article.
		Args:
			title (str): Title of wikipedia article
		Returns:
			str, The article or None, if not found
	"""
	def get(self, title):
		if title not in self.cache:
			response = requests.get(
				'https://en.wikipedia.org/w/api.php',
				params = {
					'action': 'query',
					'format': 'json',
					'titles': title,
					'prop': 'extracts',
					'explaintext': True
				}
			).json()
			page = next(iter(response['query']['pages'].values()))
			try:
				self.cache[title] = page['extract']
			except KeyError:
				self.cache[title] = None

			write_json_file(self.cachefile, self.cache)

		return self.cache[title]

	"""
		Check existence of an article.
		Args:
			title (str): Title of wikipedia article
		Returns:
			bool, does article exist?
	"""
	def exists(self, title):
		return self.get(title) != None
