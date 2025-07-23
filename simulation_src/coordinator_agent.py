from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate


class CoordinatorOutput(BaseModel):
    """Structured Coordinator Output"""
    process_id: Optional[str] = Field(default=None, description="Unique identifier for the process")
    agent_id: Optional[str] = Field(default=None, description="Unique identifier for the agent")
    agent_role: str = Field(description="Role of the agent")
    message: str = Field(description="Message for the agent")

class ListCoordinatorOutput(BaseModel):
    """Structured Coordinator Output List"""
    coordinator_outputs: List[CoordinatorOutput] = Field(description="List of coordinator outputs")

class Coordinator(BaseModel):
    """Coordinator Agent for managing processes and agents"""
    role: str = "coordinator"
    process_summary: str = Field(description="Summary of the process")
    conventions: str = Field(default="", description="Conventions to follow when assigning agents to tasks")
    llm: Any = Field(default=None, description="LLM for generating responses")

    def hand_off(self, process_states, org_agents, external_agents, current_timestamp):
        org_agents_str = ', '.join([f"ID: {agent.id} (role: {agent.role}, actions: {agent.actions})" for agent in org_agents])
        external_agents_str = ', '.join([f"ID: {agent.id} (role: {agent.role}, actions: {agent.actions})" for agent in external_agents])
        prompt = """
        You are the coordinator of a process.
        The process has many instances going on. Each instance has its unique ID and its current state.
        The current state show all the events happening so far for that instance.
        Your duty is to:
            - understand the process
            - look at the current state of all process instances
            - decide which agent should be assigned next for which process id
        You are given:
            - process_summary: the description of the process, which helps you know who do what.
            - process_states: the states of all process instances going on, which helps you know what need to be done next.
            - organization_agents: list of all agents and their roles in the organization. One agent can have multiple roles.
            - external_agents: list of all external agents and their roles
        Please output:
            - process_id: <which process to be handled next>
            - agent_id: <which agent to be assigned to that process>
            - agent_role: <the role of the agent>
            - message: <the message to the agent, what should they do for that process_id>

        If process_states is empty, it means no process is currently active. In this case, you need to call an agent to take an action that initializes the process. 
            - Simply omit the process ID in this scenario.
            - Using the process summary to decide which agent you should call to initialize the process.
            - Read the process summary carefully to identify the agent role responsible for triggering the process by performing the first action.

        If multiple processes needed to be handled and multiple agents needed to be called, output them in list.
        If you want to assign the agent within the organization, you muss define the agent_id and the agent_role.
        If you want to assign the agent outside the organization, but this agent has not existed yet in the external_agents and you need to contact a new one, define their agent_role.
    
        You must follow the conventions below when assigning agents to tasks:
        {conventions}

        ***Note***: 
            - Depending on the situation and the task, you should have suitable assignment, if the task is time consuming, you should not assign one agent with so many tasks but devide between agents who can be responsible for that task.
            - If the tasks needed to done in the sequence A->B (which means task B must be done after task A), you can't assign 2 tasks to 2 differents agents, you only can assign 2 tasks to same agent or only task A to one agent and wait until task A is done before you can assign task B.
            - If the tasks A and B can done parallel, then you can do anything of the above.
            - Don't assign different agents to same activity of the same process.
        
        Below is the information given to you:
        Process_summary: {process_summary}
        Process_states: {process_states}
        Organization_agents: {org_agents}
        External_agents: {external_agents}
        Current_timestamp: {current_timestamp}
        """

        structured_llm = self.llm.with_structured_output(ListCoordinatorOutput)
        prompt = PromptTemplate.from_template(prompt)
        chain = prompt | structured_llm
        coordinator_output = chain.invoke({
            "process_summary": self.process_summary,
            "process_states": process_states,
            "org_agents": org_agents_str,   
            "external_agents": external_agents_str,
            "current_timestamp": current_timestamp,
            'conventions':  self.conventions
        })
        return coordinator_output
        
        
        