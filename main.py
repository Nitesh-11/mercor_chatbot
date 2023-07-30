import textbase
from textbase.message import Message
from textbase import models
import os
from typing import List
from langchain.tools import BaseTool, StructuredTool, Tool, tool

from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent
from langchain import LLMMathChain
from pydantic import BaseModel, Field
# from langchain.tools import YouTubeSearchTool
# tool = YouTubeSearchTool()


class CalculatorInput(BaseModel):
    question: str = Field()
memory = ConversationBufferMemory(memory_key="chat_history")

search = GoogleSerperAPIWrapper(serper_api_key="11bcf35bd1cdba54a4875e816b1b56e5a9169bf6")

llm = OpenAI(openai_api_key="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", temperature=0)

llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [
    Tool(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    # Tool(
    #     func=tool.run,
    #     name="YoutubeLink",
    #     description="useful for finding video link for a person"
    #     # coroutine= ... <- you can specify an async method if desired as well
    # ),
    Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
]
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
# Load your OpenAI API key
models.OpenAI.api_key = "YOUR_API_KEY"
# or from environment variable:
# models.OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT = """You are chatting with an AI. There are no specific prefixes for responses, so you can ask or talk about anything you like. The AI will respond in a natural, conversational manner. Feel free to start the conversation with any question or topic, and let's have a pleasant chat!
"""


@textbase.chatbot("talking-bot")
def on_message(message_history: List[Message], state: dict = None):
    """Your chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    # # Generate GPT-3.5 Turbo response
    bot_response = agent_chain.run(message_history[-1].content)

    return bot_response, state
