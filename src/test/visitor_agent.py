# Projektpraktikum Information Retrieval Agents
# Institut für Informationssysteme der Universitaet zu Luebeck
#	Magnus Bender, 2021

import sys 

from spade.agent import Agent
from spade.behaviour import FSMBehaviour, State

STATE_PREPARATION = "STATE_PREPARATION"
STATE_AWAIT_MOIN = "STATE_AWAIT_MOIN"
STATE_END = "STATE_END"

class VisitorStateMachine(FSMBehaviour):
	async def on_start(self):
		pass

	async def on_end(self):
		await self.agent.stop()

class PreparationState(State):
	async def run(self):
		print("State: Preparation", "(Visitor " + self.agent.get("name") + ")")
		self.set_next_state(STATE_AWAIT_MOIN)

class AwaitMoinState(State):
	async def run(self):
		msg = await self.receive(timeout=sys.float_info.max)
		print("State: Await Moin, Got message '" + msg.body + "' (Visitor " + self.agent.get("name") + ")")
		self.set_next_state(STATE_END)

class EndState(State):
	async def run(self):
		print("State: END", "(Visitor " + self.agent.get("name") + ")")

class VisitorAgent(Agent):
	async def setup(self):
		asm = VisitorStateMachine()

		asm.add_state(name=STATE_PREPARATION, state=PreparationState(), initial=True)
		asm.add_state(name=STATE_AWAIT_MOIN, state=AwaitMoinState())
		asm.add_state(name=STATE_END, state=EndState())
		
		asm.add_transition(source=STATE_PREPARATION, dest=STATE_AWAIT_MOIN)
		asm.add_transition(source=STATE_AWAIT_MOIN, dest=STATE_END)

		self.add_behaviour(asm)	