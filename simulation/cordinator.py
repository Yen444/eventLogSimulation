from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate

# utils functions
from simulation.llm_utils import generate_random_text_with_mem_chatbot

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm_low = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0)
llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5)
llm_high = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)

class CordinatorOutput(BaseModel):
    """Structured Cordinator Output"""
    process_id: str = Field(description="Unique identifier for the process")
    agent_id: Optional[str] = Field(description="Unique identifier for the agent")
    agent_role: Optional[str] = Field(description="Role of the agent")
    message: str = Field(description="Message for the agent")

class ListCordinatorOutput(BaseModel):
    """Structured Cordinator Output List"""
    cordinator_outputs: List[CordinatorOutput] = Field(description="List of cordinator outputs")

class Cordinator(BaseModel):
    role: str = "cordinator"
    process_summary: str = Field(description="Summary of the process")
    llm = llm_high

    def hand_off(self, process_states, org_agents, external_agents):
        prompt = """
        You are the cordinator of a process.
        The process has many entities going on. Each entity has its unique ID and its current state.
        The current state show all the events happening so far for that entity.
        Your duty is to:
            - understand the process
            - look at the current state of all process instances
            - decide which agent should be assigned next for which process id
        You are given:
            - process_summary: the description of the process, which helps you know who do what.
            - process_states: the states of all process instances going on
            - organization_agents: list of all agents and their roles in the organization
            - external_agents: list of all external agents and their roles
        Please output:
            - process_id: <which process to be handled next>
            - agent_id: <which agent to be assigned to that process>
            - agent_role: <the role of the agent>
            - message: <if you have any message for this agent>

        If multiple processes needed to be handled. Output them in list.
        If you want to assign the agent within the organization, define the agent_id.
        If you want to assign the agent outside the organization, but this agent has not existed yet in the external_agents and you need to contact a new one, define their agent_role.
        Note: Depending on the situation and the task, you should have suitable assignment, if the task is time consuming, you should not assign one agent with so many tasks but devide between agents who can be responsible for that task.
        
        Process_summary: {process_summary}
        Process_states: {process_states}
        Organization_agents: {org_agents}
        External_agents: {external_agents}"""

        structured_llm = self.llm.with_structured_output(ListCordinatorOutput)
        prompt = PromptTemplate.from_template(self.prompt)
        chain = prompt | structured_llm
        cordinator_output = chain.invoke({
            "process_summary": self.process_summary,
            "process_states": process_states,
            "or_agents": org_agents
        })
        return cordinator_output
    
    def give_feedback(self):
        prompt = """
        You're the cordinator of a process.
        You're duty is to cordinate all the entities of the process.
        You know when a new entity (intance enters the system) and you check very often all the states of all process instances \
            and also on how the organizations are working.
        If you see any abnormal or special events happening, you can always call one of the organization agents and clarify.
        You are given:
            - process_summary: the description of the process, which helps you know who do what.
            - process_states: the states of all process instances going on
            - organization_agents: list of all agents and their roles in the organization
        Please output:
            - process_id: <which process to be handled next>
            - agent_id: <which agent to be assigned to that process>
            - agent_role: <the role of the agent>
            - message: <the clarifying message for this agent>
        If you want to assign the agent within the organization, define the agent_id.
        If you want to assign the agent outside the organization, define the agent_role.

        Process_summary: {process_summary}
        Process_states: {process_states}
        Organization_agents: {org_agents}"""
        structured_llm = self.llm.with_structured_output(CordinatorOutput)
        prompt = PromptTemplate.from_template(self.prompt)
        chain = prompt | structured_llm
        cordinator_output = chain.invoke({
            "process_summary": self.process_summary,
            "process_states": self.process_states,
            "or_agents": self.org_agents
        })
        return cordinator_output
    def answer_feedback(self, feed_back):
        prompt = """You're the coordinator of a process. You know the prcess well and you're always there to help\
            the organization agents."""
    
