import sys
import asyncio
from random import randint

from spade.agent import Agent
from spade.message import Message
from spade.behaviour import FSMBehaviour, State

import src.utils.functions as functions

STATE_PREPARATION = "STATE_PREPARATION"
# Added states
STATE_SEND_QUERY = "STATE_SEND_QUERY"
STATE_AWAIT_ANSWER = "STATE_AWAIT_ANSWER"
# End added states
STATE_END = "STATE_END"


class QuestionerStateMachine(FSMBehaviour):
    async def on_start(self):
        print("Starting query")

    async def on_end(self):
        print("Ending query")
        await self.agent.stop()


class PreparationState(State):
    async def run(self):
        """ For debugging purpose or to keep track on the current state:
        print("State: Preparation", self.agent.name) """
        query_list = functions.read_file("./exercise3/data/" + self.agent.get("queries_file") + ".txt")

        query_list = query_list.split("\n")
        # change to smaller subset
        self.agent.set("query_list", query_list[1:5])
        # Move to next state
        self.set_next_state(STATE_SEND_QUERY)


class SendQueryState(State):
    async def run(self):
        """ For debugging purpose or to keep track on the current state:
        print("State: Send Query", self.agent.name) """
        if not self.agent.get("query_list"):
            # Move to End-state if there are no more queries left
            self.set_next_state(STATE_END)
        else:
            # Get query list
            query_list = self.agent.get("query_list")
            # Get random query element from list
            query = query_list.pop(randint(0, len(query_list)-1))
            self.agent.set("query", query)
            msg = Message(to=self.agent.get("query_target"))
            # Send message to IR Agent with query and location of tokenized query
            msg.body = query
            await self.send(msg)
            # Move to next state
            self.set_next_state(STATE_AWAIT_ANSWER)


class AwaitAnswerState(State):
    async def run(self):
        """ For debugging purpose or to keep track on the current state:
        print("State: Await answer", self.agent.name) """
        msg = await self.receive(timeout=sys.float_info.max)
        # Print answer
        print("For Agent", self.agent.name, "with query", self.agent.get("query"), "\n")
        print("Got:", "\n", msg.body)
        # Possible wait for a random time
        rand = 1 #randint(0, 5)
        for i in range(0, rand):
            await asyncio.sleep(0)
        # Move to next state
        self.set_next_state(STATE_SEND_QUERY)


class EndState(State):
    async def run(self):
        msg = Message(to=self.agent.get("query_target"))
        msg.body = "Ending query."
        await self.send(msg)
        # Giving the cpu to other agents, so they can end properly
        await asyncio.sleep(0)


class QuestionerAgent(Agent):
    async def setup(self):
        asm = QuestionerStateMachine()

        asm.add_state(name=STATE_PREPARATION, state=PreparationState(), initial=True)
        # Added states
        asm.add_state(name=STATE_SEND_QUERY, state=SendQueryState())
        asm.add_state(name=STATE_AWAIT_ANSWER, state=AwaitAnswerState())
        # Added states
        asm.add_state(name=STATE_END, state=EndState())

        # adding transitions
        asm.add_transition(source=STATE_PREPARATION, dest=STATE_SEND_QUERY)
        # Added transitions
        asm.add_transition(source=STATE_SEND_QUERY, dest=STATE_AWAIT_ANSWER)
        asm.add_transition(source=STATE_AWAIT_ANSWER, dest=STATE_SEND_QUERY)
        # Added transitions
        asm.add_transition(source=STATE_SEND_QUERY, dest=STATE_END)

        self.add_behaviour(asm)
