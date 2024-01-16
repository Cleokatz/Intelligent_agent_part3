import sys
from spade.agent import Agent
from spade.message import Message
from spade.behaviour import FSMBehaviour, State
import src.utils.functions as functions

from random import randint

STATE_PREPARATION = "STATE_PREPARATION"
# Added states
STATE_GIVE_DOCUMENT = "STATE_GIVE_DOCUMENT"
STATE_AWAIT_BID = "STATE_AWAIT_BID"
STATE_SAY_WINNER = "STATE_SAY_WINNER"
# End added states
STATE_END = "STATE_END"


# Set behaviour of auctioneer on start and end, create finite state machine agent
class AuctioneerStateMachine(FSMBehaviour):
    async def on_start(self):
        print("Starting auction")

    async def on_end(self):
        print("Ending auction")
        await self.agent.stop()


# State of auctioneer to prepare for the auction: Load the list with the documents to sell, only called once
class PreparationState(State):
    async def run(self):
        """ For debugging purpose or to keep track on the current state:
        print("State: Preparation, load sell documents") """
        # Load document as string
        sell_list = functions.read_file("./exercise3/data/" + self.agent.get("documents_file") + ".txt")
        # Change string into list
        sell_list = sell_list.split("\n")[1:5]
        # Store sellDocuments in the knowledge base of agent, can be used for a smaller set of documents
        self.agent.set("sellDocuments", sell_list)
        print(sell_list)
        # Move to next state
        self.set_next_state(STATE_GIVE_DOCUMENT)


# State of auctioneer to give one document to sell to the bidder. The state is beginning of a loop
class GiveDocState(State):
    async def run(self):
        """ For debugging purpose or to keep track on the current state:
        print("State: Give Document to bidder") """
        # Get the list to sell the documents
        sell_list = self.agent.get("sellDocuments")
        # Select the document to sell randomly, so there is no structure anymore
        sell_item = sell_list.pop(randint(0, len(sell_list) - 1))
        # Store sell_item in the knowledge base of agent
        self.agent.set("sell_item", sell_item)
        # Send message to the bidder with the title of the document to sell
        for bidder in self.agent.get("bidders_list"):
            msg = Message(to=bidder)
            # Only the title, not entire doc is sent to the bidder
            msg.body = "The document for sale is '" + str(sell_item) + "'"

            await self.send(msg)
        # Move to next state
        self.set_next_state(STATE_AWAIT_BID)


# State of auctioneer to wait for the bids of the bidder. Calculates the winner on the given bids
class AwaitBidState(State):
    async def run(self):
        """ For debugging purpose or to keep track on the current state:
        print("State: Await Bids from both") """
        # Awaits the messages from the bidder, in this auction the bidder have to send a message,
        # even if they don't want to bid on this item, so there is no need of a specific timeout after n seconds
        list_bids_bidder = []
        for bidder in self.agent.get("bidders_list"):
            # Receive message of bidder
            msg = await self.receive(timeout=sys.float_info.max)
            # Store name and amount of bidder
            list_bids_bidder.append((str(msg.sender).split("@")[0], float(msg.body)))
        # Optional print to see how much then bidder are willing to pay
        # print("State: Got bids:", list_bids_bidder)
        # Sort list after biggest value
        list_bids_bidder.sort(key=lambda bid: bid[1], reverse=True)
        # If both bidder bids nothing, the document gets ejected
        # If they give identical bids, it is randomly chosen who is winner, can be adapted to more than two bidder
        if list_bids_bidder[0][1] == 0.0:
            # Tell the bidder that this document is ejected
            for bidder in self.agent.get("bidders_list"):
                msg = Message(to=bidder)
                msg.body = "No bids for the documents. Eject this. Going to the next one."
                await self.send(msg)
            # List of selling documents is empty -> end auction
            if not self.agent.get("sellDocuments"):
                self.set_next_state(STATE_END)
            # List of selling documents not empty -> Sell next document
            else:
                self.set_next_state(STATE_GIVE_DOCUMENT)
        elif list_bids_bidder[0][1] == list_bids_bidder[1][1]:
            chance = randint(0, 1)
            self.agent.set("winner", list_bids_bidder[chance])
            # Move to next state
            self.set_next_state(STATE_SAY_WINNER)
        else:
            self.agent.set("winner", list_bids_bidder[0])
            # Move to next state
            self.set_next_state(STATE_SAY_WINNER)


# State of auctioneer to say the winner to the bidder. Checks if there are more documents to sell. End of the loop
class SayWinnerState(State):
    async def run(self):
        """ For debugging purpose or to keep track on the current state:
        print("State: Say Winner") """
        # Can be deleted, print for seeing which bidder is the winner for which document
        print("Winner is:" + self.agent.get("winner")[0], "for the Document:", str(self.agent.get("sell_item")))
        # Send winner to the bidder
        for bidder in self.agent.get("bidders_list"):
            msg = Message(to=bidder)
            msg.body = "The Winner of the document " + str(self.agent.get("sell_item")) + \
                       " is " + str(self.agent.get("winner")[0])
            await self.send(msg)
        # List of selling documents is empty -> end auction
        if not self.agent.get("sellDocuments"):
            self.set_next_state(STATE_END)
        # List of selling documents not empty -> Sell next document
        else:
            self.set_next_state(STATE_GIVE_DOCUMENT)


# State of auctioneer to end auction by sending the message to the bidder.
class EndState(State):
    async def run(self):
        for bidder in self.agent.get("bidders_list"):
            msg = Message(to=bidder)
            msg.body = "No more documents to sell. Closing Auction!"
            await self.send(msg)


# Defines states and transitions of the auctioneer agent
class AuctioneerAgent(Agent):
    async def setup(self):
        asm = AuctioneerStateMachine()

        asm.add_state(name=STATE_PREPARATION, state=PreparationState(), initial=True)
        # Added states
        asm.add_state(name=STATE_GIVE_DOCUMENT, state=GiveDocState())
        asm.add_state(name=STATE_AWAIT_BID, state=AwaitBidState())
        asm.add_state(name=STATE_SAY_WINNER, state=SayWinnerState())
        # Added states
        asm.add_state(name=STATE_END, state=EndState())

        # adding transitions
        asm.add_transition(source=STATE_PREPARATION, dest=STATE_GIVE_DOCUMENT)
        # Added transitions
        asm.add_transition(source=STATE_GIVE_DOCUMENT, dest=STATE_AWAIT_BID)
        asm.add_transition(source=STATE_AWAIT_BID, dest=STATE_GIVE_DOCUMENT)
        asm.add_transition(source=STATE_AWAIT_BID, dest=STATE_SAY_WINNER)
        asm.add_transition(source=STATE_SAY_WINNER, dest=STATE_GIVE_DOCUMENT)
        # Added transitions
        asm.add_transition(source=STATE_AWAIT_BID, dest=STATE_END)
        asm.add_transition(source=STATE_SAY_WINNER, dest=STATE_END)

        self.add_behaviour(asm)
