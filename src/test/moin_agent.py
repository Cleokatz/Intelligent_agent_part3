# Projektpraktikum Information Retrieval Agents
# Institut für Informationssysteme der Universitaet zu Luebeck
#	Magnus Bender, 2021

from spade.agent import Agent
from spade.message import Message
from spade.behaviour import FSMBehaviour, State

STATE_PREPARATION = "STATE_PREPARATION"
STATE_SAY_MOIN = "STATE_SAY_MOIN"
STATE_END = "STATE_END"

class MoinStateMachine(FSMBehaviour):
	async def on_start(self):
		pass

	async def on_end(self):
		await self.agent.stop()

class PreparationState(State):
	async def run(self):
		print("State: Preparation (Moin)")
		self.set_next_state(STATE_SAY_MOIN)
		
class SayMoinState(State):
	async def run(self):
		print("State: Moin (Moin)")

		for count,visitor in enumerate(self.agent.get("visitors")):
			msg = Message(to=visitor)
			msg.body = "Moin Visitor" + str(count)

			await self.send(msg)
		
		self.set_next_state(STATE_END)

class EndState(State):
	async def run(self):
		print("State: End (Moin)")

class MoinAgent(Agent):
	async def setup(self):
		asm = MoinStateMachine()

		asm.add_state(name=STATE_PREPARATION, state=PreparationState(), initial=True)
		asm.add_state(name=STATE_SAY_MOIN, state=SayMoinState())
		asm.add_state(name=STATE_END, state=EndState())

		asm.add_transition(source=STATE_PREPARATION, dest=STATE_SAY_MOIN)
		asm.add_transition(source=STATE_SAY_MOIN, dest=STATE_END)

		self.add_behaviour(asm)	