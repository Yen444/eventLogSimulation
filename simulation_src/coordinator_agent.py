from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

import json


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
    id: str = Field(description="Unique identifier for the coordinator")
    role: str = "coordinator"
    process_summary: str = Field(description="Summary of the process")
    conventions: str = Field(default="", description="Conventions to follow when assigning agents to tasks")
    llm: Any = Field(default=None, description="LLM for generating responses")
    handoff_fewshots: List[Dict[str, str]] = Field(default_factory=list, description="Few-shot examples for handoff")

    def hand_off(self, process_states, org_agents, external_agents, current_timestamp):
        org_agents_str = '\n'.join([f"ID: {agent.id}; (role: {agent.role}; actions: {agent.actions}; data: {agent.data})" for agent in org_agents])
        external_agents_str = '\n'.join([f"ID: {agent.id}; (role: {agent.role}; actions: {agent.actions}; data: {agent.data})" for agent in external_agents])
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
            **Individual Process Evaluation is Paramount:** Each task assignment *must* be determined by the specific, current state of its *individual process*. Never batch tasks across processes without verifying each process's unique state; the required task for a process is *solely dependent on that process's current state*.

            **Optimize Resource Allocation:** For time-consuming tasks, distribute the workload among multiple capable agents to prevent bottlenecks and ensure efficient completion.

            **Respect Task Dependencies:**
                - If Task B must follow Task A for the same process (A -> B), either assign both to the *same agent*, or assign Task A, *wait for its completion and result*, then assign Task B.
                - Always wait for a preceding task's completion and result before assigning any dependent subsequent tasks.
                - Leverage parallelism: Tasks within the same process that can run in parallel, or tasks belonging to different processes, can be assigned to different agents or the same agent concurrently.

            **Maintain Process Integrity:**
                - Never assign different agents to the same activity within the same process.
                - Strictly adhere to Process IDs: Each process has a unique state and guidelines. Carefully read Process IDs and their current states to ensure every assignment aligns precisely with that process's specific requirements and general guidelines.

        Below is the information given to you:
        Process_summary: {process_summary}
        Process_states: {process_states}
        Organization_agents: {org_agents}
        External_agents: {external_agents}
        Current_timestamp: {current_timestamp}

        Below are some examples of how to assign agents to tasks basing on given inputs:
        """
        fewshot_examples = self.handoff_fewshots
        if fewshot_examples:
            # example prompt template
            example_prompt = PromptTemplate(
                input_variables=["process_summary", "process_states", "organization_agents", "external_agents", "current_timestamp", "conventions", "output"],
                template="Process_summary: {process_summary}\nProcess_states: {process_states}\nOrganization_agents: {organization_agents}\nExternal_agents: {external_agents}\nCurrent_timestamp: {current_timestamp}\nConventions: {conventions}\nOutput: {output}"
            )
            final_prompt = FewShotPromptTemplate(
                examples=fewshot_examples,
                example_prompt=example_prompt,
                input_variables=["process_summary", "process_states", "org_agents", "external_agents", "current_timestamp", "conventions"],
                prefix=prompt
            )

        else:
            final_prompt = PromptTemplate.from_template(prompt)

        structured_llm = self.llm.with_structured_output(ListCoordinatorOutput) #
        chain = final_prompt | self.llm
        raw_output = chain.invoke({
            "process_summary": self.process_summary,
            "process_states": process_states,
            "org_agents": org_agents_str,   
            "external_agents": external_agents_str,
            "current_timestamp": current_timestamp,
            'conventions':  self.conventions
        })
        coordinator_output = structured_llm.invoke(raw_output.content)
        return raw_output, coordinator_output
        
        
        