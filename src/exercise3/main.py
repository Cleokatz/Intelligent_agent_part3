#!/usr/bin/env python3

from src.utils.random import Random
import spade

from src.utils import helpfunctions
from src.exercise3.agent_ir import IRAgent
from src.exercise3.agent_auctioneer import AuctioneerAgent
from src.exercise3.agent_questioner import QuestionerAgent
initialized = True  # For downloading in json
preprocessed = True  # For preprocessing either with bigrams (lda=True) or without
gen_lda_model = True  # Create lda_model

async def main():
	if not initialized:
		helpfunctions.download_wiki()
	if not preprocessed:
		helpfunctions.preprocessing_all(lda_preprocessing=True)
	if not gen_lda_model:
		lda_model = None  # gensim.models.LdaMulticore.load("./model/lda_model_corpus")
		helpfunctions.make_lda(lda_model)
		return

	# Randomly choose a set of queries for Questioner
	generator = Random.get_generator()
	all_queries = ["queries_i", "queries_ii", "queries_iii", "queries_iv", "queries_v", "queries_vi"]
	queryset_1, queryset_2 = generator.sample(all_queries, 2)

	# IR agents
	ir1 = IRAgent("bidder1@localhost", "bidder1")
	ir1.set("corpus_file", "corpus")
	ir1.set("strategy", "reward_orient") # standard
	ir1.set("competitor_sets", [q for q in all_queries if q != queryset_1])
	ir1.set("competitor_sets_all", [q for q in all_queries if q != queryset_1])
	ir1.set("auctioneer", "auctioneer@localhost")
	await ir1.start()

	ir2 = IRAgent("bidder2@localhost", "bidder2")
	ir2.set("corpus_file", "corpus")
	ir2.set("strategy", "reward_orient") # standard
	ir2.set("competitor_sets", [q for q in all_queries if q != queryset_2])
	ir2.set("competitor_sets_all", [q for q in all_queries if q != queryset_2])
	ir2.set("auctioneer", "auctioneer@localhost")
	await ir2.start()

	# Auctioneer
	auctioneer = AuctioneerAgent("auctioneer@localhost", "auctioneer")
	auctioneer.set("bidders_list", ['bidder1@localhost', 'bidder2@localhost'])
	auctioneer.set("documents_file", "sell")
	await auctioneer.start()

	# Questioners
	questioner1 = QuestionerAgent("questioner1@localhost", "questioner1")
	questioner1.set("query_target", "bidder1@localhost")
	questioner1.set("queries_file", queryset_1)
	await questioner1.start()

	questioner2 = QuestionerAgent("questioner2@localhost", "questioner2")
	questioner2.set("query_target", "bidder2@localhost")
	questioner2.set("queries_file", queryset_2)
	await questioner2.start()
	ir1.set("queries_file", queryset_1)
	ir1.set("opp", "bidder2@localhost")
	ir2.set("queries_file", queryset_2)
	ir2.set("opp", "bidder1@localhost")
if __name__ == "__main__":
	spade.run(main())
