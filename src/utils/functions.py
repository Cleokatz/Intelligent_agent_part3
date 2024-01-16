# Projektpraktikum Information Retrieval Agents
# Institut für Informationssysteme der Universitaet zu Luebeck
#	Magnus Bender, 2021

import re, json, os

from src.utils.external import NumpyEncoder

def clear_filename(s):
	'''
		Clear a string to use as filename
	'''
	return re.sub(r'[^0-9a-zA-Z_\-]', '', s.replace(' ', '_'))

def read_file(filename):
	'''
		Read file and return content as string
	'''
	content = ""
	with open(filename, 'r', errors='ignore') as f:
		for line in f:
			line = line.strip()
			if len(line) > 0:
				content += "\n" + line
	return content

def read_json_file(filename, errors=None, encoding=None):
	'''
		Read a file and return contained json-object 
	'''
	return json.load(open(filename, "r", errors=errors, encoding=encoding))

def write_file(filename, content):
	'''
		Write content (= string) to file
	'''
	f = open(filename, "w")
	f.write(content)
	f.close()

def write_json_file(filename, content):
	'''
		Write content to file, but encode via json (so dicts, lists may be dumped to file)
	'''
	write_file(filename, json.dumps(content, indent=2, cls=NumpyEncoder))

def check_and_create_folder(path):
	'''
		Make sure that the given path/folder (=string) exists (will create if not)
	'''
	if not os.path.isdir(path):
		os.mkdir(path)
