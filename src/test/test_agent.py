#!/usr/bin/env python3

# Projektpraktikum Information Retrieval Agents
# Institut für Informationssysteme der Universitaet zu Luebeck
#	Magnus Bender, 2021

import spade

from src.test.moin_agent import MoinAgent
from src.test.visitor_agent import VisitorAgent
# if used outside Docker/ Python package, place all files in one folder and use lines below
#from moin_agent import MoinAgent
#from visitor_agent import VisitorAgent

async def main():

	# create two visitors (each awaiting a "Moin")
	visitor1 = VisitorAgent("bidder1@localhost", "bidder1")
	visitor1.set("name", "Otto")
	await visitor1.start()

	visitor2 = VisitorAgent("bidder2@localhost", "bidder2")
	visitor2.set("name", "Heinz")
	await visitor2.start()

	# create one agent saying "Moin" to all visitors
	moin = MoinAgent("auctioneer@localhost", "auctioneer")
	moin.set("visitors", ['bidder1@localhost', 'bidder2@localhost'])
	await moin.start()


if __name__ == "__main__":
	spade.run(main())
