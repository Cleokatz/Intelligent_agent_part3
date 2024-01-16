import sys 
import asyncio

from gensim.matutils import cossim
from gensim.models import TfidfModel
from spade.agent import Agent
from spade.message import Message
from spade.behaviour import FSMBehaviour, State
import src.utils.functions as functions
import src.utils.helpfunctions as helpfunc
import gensim
import os

STATE_PREPARATION = "STATE_PREPARATION"
# Added states
STATE_AWAIT_DOC = "STATE_AWAIT_DOC"
STATE_AWAIT_WIN = "STATE_AWAIT_WIN"
# Added states
STATE_END = "STATE_END"


# Set behaviour of information retrieval agent on start and end, create finite state machine agent
class IRStateMachine(FSMBehaviour):
	async def on_start(self):
		print("Start bidding (", self.agent.name, ")")

	async def on_end(self):
		print("End bidding (", self.agent.name, ")")
		await self.agent.stop()


# State of information retrieval agent to prepare for the auction: load and preprocess corpus
class PreparationState(State):
	async def run(self):
		# Set some important parameters
		self.agent.set("msg_list", [])
		self.agent.set("auction", True)
		self.agent.set("ir", False)#True
		self.agent.set("query", ["Aston Martin"])
		self.agent.set("got_query", False)
		# Load dictionary and database
		dct, database = helpfunc.create_dictionaries()
		self.agent.set("dct_database", (dct, database))
		# Set corpus_list
		# TODO: Als klasse mit für gegner?
		corpus_list = functions.read_file("./exercise2/data/corpus.txt")
		corpus_list = corpus_list.split("\n")[1:]
		self.agent.set("corpus_list", corpus_list)
		self.agent.set("oppo_corpus_list", corpus_list)
		# Load the lda_model
		#lda_model = gensim.models.LdaMulticore.load("./model/lda_model_corpus")
		#self.agent.set("lda_model", lda_model)

		# Move to next state
		self.set_next_state(STATE_AWAIT_DOC)

class AwaitDocState(State):
	async def run(self):
		""" For debugging purpose or to keep track on the current state:
		print("Await Doc State") """
		# Decision which process to start by looking which message was received
		if not self.agent.get("msg_list") == []:
			element = self.agent.get("msg_list").pop()
			await self.info_retrieval(element)
		else:
			msg = await self.receive(timeout=sys.float_info.max)
			# No query so the ir-agent can not answer the auction -> Wait for query
			if self.agent.get("query") == [] and str(msg.sender) == self.agent.get("auctioneer"):
				msg_auctioneer = msg
				msg = await self.receive(timeout=sys.float_info.max)
				await self.info_retrieval(msg)
				await self.auction(msg_auctioneer)
			# Start auction
			elif str(msg.sender) == self.agent.get("auctioneer"):
				await self.auction(msg)
			# Wait for auction if we already become a query
			elif len(self.agent.get("query")) >= 1 and self.agent.get("auction") and not str(msg.sender) == self.agent.get("auctioneer"):
				while not str(msg.sender) == self.agent.get("auctioneer"):
					self.agent.get("msg_list").append(msg)
					msg = await self.receive(timeout=sys.float_info.max)
				await self.auction(msg)
			# Make information_retrieval
			else:
				await self.info_retrieval(msg)

	# Process to handle information retrieval and answer the queries from the questioner
	async def info_retrieval(self, msg):
		return
		# If both ends: Goto end_state, else make the ir part false
		if "Ending query" in msg.body:
			self.agent.set("ir", False)
			if not self.agent.get("auction"):
				self.set_next_state(STATE_END)
			else:
				self.set_next_state(STATE_AWAIT_DOC)
		else:
			self.agent.get("query").append(msg.body)
			lda_model = self.agent.get("lda_model")
			distances = helpfunc.get_distances(corpus_titles=self.agent.get("corpus_list"), query_title=msg.body, lda_model=lda_model, database_all=self.agent.get("database_tokenized"))
			# print(distances)
			documents = helpfunc.get_discrepancy(distances, self.agent.get("corpus_list"), self.agent.get("database_tokenized"), msg.body)
			msg = msg.make_reply()  # TODO: better human answering --> komplette anpassung der Nachricht
			msg.body = "Titel of some wonderful documents: " + str(documents)
			await self.send(msg)
			self.set_next_state(STATE_AWAIT_DOC)

	# Process to handle the auction
	async def auction(self, msg):
		# If both ends: Goto end_state, else make the auction part false
		if "Closing Auction" in msg.body:
			self.agent.set("auction", False)
			if not self.agent.get("ir"):
				self.set_next_state(STATE_END)
			else:
				self.set_next_state(STATE_AWAIT_DOC)
		else:
			# Filter out the selling document
			buy_doc = str(msg.body).split("'")[1]
			self.agent.set("buy_doc", buy_doc)
			# Append the document to the corpus_list
			self.agent.get("corpus_list").append(buy_doc)
			self.agent.get("oppo_corpus_list").append(buy_doc)
			if buy_doc in self.agent.get("query"):
				price = 10
			else:
				# Calculate the tf_idf on the corpus with the new document
				dct, database = self.agent.get("dct_database")
				score = helpfunc.calculate_score(dct, database, self.agent.get("corpus_list"), self.agent.get("query"), buy_doc)
				opponent_score = helpfunc.calculate_score(dct, database, self.agent.get("oppo_corpus_list"), self.agent.get("oppo_query"), buy_doc)
				price = helpfunc.get_value(score, opponent_score)
			# Send message with the price to the auctioneer
			msg = Message(to=self.agent.get("auctioneer"))
			msg.body = str(price)

			await self.send(msg)

			# Move to next state
			self.set_next_state(STATE_AWAIT_WIN)


# State of bidder to get the announcement of the winner. If the bidder is the winner, the document gets in his corpus
class AwaitWinState(State):
	async def run(self):
		""" For debugging purpose or to keep track on the current state:
		print("Await win state")"""
		# Wait for announcement of the winner
		msg = await self.receive(timeout=sys.float_info.max)
		while not str(msg.sender) == self.agent.get("auctioneer"):
			self.agent.get("msg_list").append(msg)
			msg = await self.receive(timeout=sys.float_info.max)
		# The bidder not the winner -> remove document from the corpus_list
		if self.agent.name not in msg.body:
			self.agent.get("corpus_list").pop()
			# TODO: Gegnerbased update
			if not self.agent.get("know_oppo_query"):
				pass # ermitteln der gegnerischen query
			dct, database = self.agent.get("dct_database")
			lda_model = helpfunc.create_new_model(dct, database, self.agent.get("oppo_corpus_list"))
			self.agent.set("oppo_lda_model", lda_model)
		else:
			# Else update the lda model with the new bought document
			pass
			self.agent.get("oppo_corpus_list").pop()
			dct, database = self.agent.get("dct_database")
			lda_model = helpfunc.create_new_model(dct, database, self.agent.get("corpus_list"))
			self.agent.set("lda_model", lda_model)
		# Move to next state
		self.set_next_state(STATE_AWAIT_DOC)


# State of bidder to end bidding
class EndState(State):
	async def run(self):
		""" For debugging purpose or to keep track on the current state:
		print("End state: Goodbye to auction") """
		print("End state (", self.agent.name, ")")

class IRAgent(Agent):
	async def setup(self):
		bsm = IRStateMachine()

		bsm.add_state(name=STATE_PREPARATION, state=PreparationState(), initial=True)
		# Added states
		bsm.add_state(name=STATE_AWAIT_DOC, state=AwaitDocState())
		bsm.add_state(name=STATE_AWAIT_WIN, state=AwaitWinState())
		# Added states
		bsm.add_state(name=STATE_END, state=EndState())

		bsm.add_transition(source=STATE_PREPARATION, dest=STATE_AWAIT_DOC)
		# Added transitions
		bsm.add_transition(source=STATE_AWAIT_DOC, dest=STATE_AWAIT_WIN)
		bsm.add_transition(source=STATE_AWAIT_WIN, dest=STATE_AWAIT_DOC)
		bsm.add_transition(source=STATE_AWAIT_DOC, dest=STATE_AWAIT_DOC)
		# Added transitions
		bsm.add_transition(source=STATE_AWAIT_DOC, dest=STATE_END)
		
		self.add_behaviour(bsm)