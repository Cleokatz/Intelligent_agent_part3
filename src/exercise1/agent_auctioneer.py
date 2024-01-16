import sys 

from spade.agent import Agent
from spade.message import Message
from spade.behaviour import FSMBehaviour, State

STATE_PREPARATION = "STATE_PREPARATION"
# TODO -- add states
STATE_END = "STATE_END"

class AuctioneerStateMachine(FSMBehaviour):
	async def on_start(self):
		pass

	async def on_end(self):
		await self.agent.stop()

class PreparationState(State):
	async def run(self):
		pass

# TODO -- add states and functionality

class EndState(State):
	async def run(self):
		pass

class AuctioneerAgent(Agent):
	async def setup(self):
		asm = AuctioneerStateMachine()

		asm.add_state(name=STATE_PREPARATION, state=PreparationState(), initial=True)
		# TODO -- add states 
		asm.add_state(name=STATE_END, state=EndState())
		
		# adding transitions
		asm.add_transition(source=STATE_PREPARATION, dest='some-state')
		# TODO -- add state transitions
		asm.add_transition(source='some-state', dest=STATE_END)
		
		self.add_behaviour(asm)	