#!/usr/bin/env python3

import spade

from src.exercise1.agent_bidder import BidderAgent
from src.exercise1.agent_auctioneer import AuctioneerAgent

async def main():
	bidder1 = BidderAgent("bidder1@localhost", "bidder1")
	await bidder1.start()

	bidder2 = BidderAgent("bidder2@localhost", "bidder2")
	await bidder2.start()

	auctioneer = AuctioneerAgent("auctioneer@localhost", "auctioneer")
	# TODO -- e.g. set a list of bidders
	# 	auctioneer.set("bidders_list", ['bidder1@localhost', 'bidder2@localhost'])
	await auctioneer.start()

if __name__ == "__main__":
	spade.run(main())
