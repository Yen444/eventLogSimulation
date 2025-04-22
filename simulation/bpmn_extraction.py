from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate

from simulation.utils import read_bpmn_file
import json

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)

### Structured output for LLM extracting BPMN role and action

class RoleAction(BaseModel):
    """Structured output for LLM extracting BPMN role and action"""
    role: str = Field(
        description="The name of the process participant (organizational role)"
    )
    actions: List[str] = Field(
        description="A list of actions corresponding to the role"
    )
    @field_validator('actions')
    @classmethod
    def validate_actions(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise ValueError("actions must be a list of strings")
        return v
    
class ListRoleAction(BaseModel):
    """Structured output for LLM extracting BPMN role and action"""
    role_action_list: List[RoleAction] = Field(
        description="A list of RoleAction objects, each containing a role and its actions"
    )

    @field_validator('role_action_list')
    @classmethod
    def validate_role_actions(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, RoleAction) for item in v):
            raise ValueError("role_action_list must be a list of RoleAction objects")
        return v
    
### Structured output for LLM extracting BPMN start and end events

class StartEndEvents(BaseModel):
    """Structured output for LLM extracting BPMN start and end events"""
    start_events: List[str] = Field(
        description="A list of start events in the BPMN model"
    )
    first_activities: List[str] = Field(
        description="A list of activities that come immediately after each start event"
    )
    end_events: List[str] = Field(
        description="A list of end events in the BPMN model"
    )
    last_activities: List[str] = Field(
        description="A list of activities that come immediately before each end event"
    )

    @field_validator('start_events', 'first_activities', 'end_events', 'last_activities')
    @classmethod
    def validate_events(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise ValueError("events must be a list of strings")
        return v

### Structured output for LLM extracting BPMN activity and process state mapping

class ActivityProcessState(BaseModel):
    """Structured output for LLM extracting BPMN activity and process state"""
    activity: str = Field(
        description="The name of the BPMN activity"
    )
    process_state: str = Field(
        description="The corresponding process state for the activity"
    )
    @field_validator('activity', 'process_state')
    @classmethod
    def validate_activity_process_state(cls, v):
        if not isinstance(v, str):
            raise ValueError("activity and process_state must be strings")
        return v
    
class ListActivityProcessState(BaseModel):
    """Structured output for LLM extracting BPMN activity and process state"""
    activity_process_states: List[ActivityProcessState] = Field(
        description="A list of ActivityProcessState objects, each containing an activity and its corresponding process state"
    )

    @field_validator('activity_process_states')
    @classmethod
    def validate_activity_process_states(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, ActivityProcessState) for item in v):
            raise ValueError("activity_process_states must be a list of ActivityProcessState objects")
        return v

### Structured output for LLM extracting process summary
class ProcessSummary(BaseModel):
    """Structured output for LLM extracting process summary"""
    process_sumary: str = Field(
        description="A short summary of the BPMN process")
    
########### BPMNExtractor class ########### 
# This class is responsible for extracting all necessary information from the BPMN model, which are needed for the simulation.
class BPMNExtractor():
    def __init__(self, bpmn_path: str, llm = llm):
        self.llm = llm
        self.model_string = read_bpmn_file(bpmn_path)
    
    def extract_process_summary(self)-> ProcessSummary:
        """Extracts a short summary of the BPMN process."""
        # Define the system prompt for the LLM
        sys_prompt = """
        You are an expert in business process modeling, tasked with analyzing BPMN diagrams represented in XML (.bpmn files). Given a BPMN model:

        ```xml
        {model}
        
        Your task is to provide a short summary of the process represented in the BPMN model. The summary should be concise and informative, capturing the main purpose and flow of the process.

        Instructions:
        - Avoid technical jargon and keep the language simple and clear.
        - Limit your summary to no more than 200 words.
        """

        prompt = PromptTemplate.from_template(sys_prompt)
        
        # Use the Pydantic model directly, not as a dictionary
        structured_llm = llm.with_structured_output(ProcessSummary)

        # build the chain with the prompt and the structured LLM
        chain = prompt | structured_llm
        
        ## Invoke the LLM with the model string
        process_summary = chain.invoke({'model': self.model_string})
        
        return process_summary

    def extract_role_actions(self)-> ListRoleAction:
        """Extracts process participants and their action spaces from the BPMN model."""

        # Define the system prompt for the LLM
        sys_prompt = """
        You are an expert in business process modeling, tasked with analyzing BPMN diagrams represented in XML (.bpmn files).  Given a BPMN model:

        ```xml
        {model}
        
        Identify all potential process participants (i.e., organizational roles) and their corresponding action spaces (the activities they perform within the process).

        Definitions:
        - Process participants: Typically roles such as "customer," "manager," or "clerk."
        - Action spaces: The set of activities (e.g., "submit request," "review application") performed by each role.

        Instructions:
        - If the BPMN diagram uses lanes and explicitly defines roles, extract the roles and their associated actions directly.
        - If roles are not explicitly defined, infer them based on task names and process structure. Use concise and intuitive names for inferred roles.
        - Prioritize clarity and conciseness in your output.

        Return your analysis in the following format for each role:
        - role: <role name>
        - actions: [<action1>, <action2>, ...]
        
        List each role and its corresponding actions separately using the format above.
        """

        prompt = PromptTemplate.from_template(sys_prompt)
        
        # Use the Pydantic model directly, not as a dictionary
        structured_llm = llm.with_structured_output(ListRoleAction)
        
        # build the chain with the prompt and the structured LLM
        chain = prompt | structured_llm
        
        ## Invoke the LLM with the model string
        role_act_list = chain.invoke({'model': self.model_string})
        return role_act_list
    
    def extract_start_end_events(self)-> StartEndEvents:
        """Extracts start/end events and first/last activities from the BPMN model."""
        # Define the system prompt for the LLM
        sys_prompt = """
        You are an expert in business process modeling, tasked with analyzing BPMN diagrams represented in XML (.bpmn files). Given a BPMN model:

        ```xml
        {model}
        
        Your task is to extract:
        - start_events: Events that trigger the start of a process.
        - first_activities: names of the activities that comes immediately after each start event.
        - end_events: Events that signify the completion of a process.
        - last_activities: names of the activities that comes immediately before each end event.

        Instructions:
        1. Identify all start and end events by their tags.
        2. Trace outgoing sequence flows from start events to find the first activities.
        3. Trace incoming sequence flows to end events to find the last activities.
        4. If the start and end event has `name` attribute, use it as the event name. Otherwise, use the tag name (e.g., "startEvent" or "endEvent").
        5. The activities always have a `name` attribute, so use it as the activity name.

        Return your analysis in the following format:
        - start_events: [<start_event1>, <start_event2>, ...]
        - first_activities: [<first_activity1>, <first_activity2>, ...]
        - end_events: [<end_event1>, <end_event2>, ...]
        - last_activities: [<last_activity1>, <last_activity2>, ...]
        """

        prompt = PromptTemplate.from_template(sys_prompt)
        
        # Use the Pydantic model directly, not as a dictionary
        structured_llm = llm.with_structured_output(StartEndEvents)

        # build the chain with the prompt and the structured LLM
        chain = prompt | structured_llm
        
        ## Invoke the LLM with the model string
        start_end_events = chain.invoke({'model': self.model_string})

        return start_end_events

    def extract_act_states(self)-> ListActivityProcessState:
        """Extracts the mapping between activities and their corresponding process states from the BPMN model."""

        # Define the system prompt for the LLM
        sys_prompt = """
        You are an expert in business process modeling, tasked with analyzing BPMN diagrams represented in XML (.bpmn files). Given a BPMN model:

        ```xml
        {model}
        
        Identify all activities and their corresponding process states in the BPMN model.

        Definitions:
        - Activities: Tasks or actions performed within the process.
        - Process states: The states or conditions that result from executing activities.

        Instructions:
        - The activities are represented as tasks in the BPMN model.
        - The process states are the conditions or states that result from executing the activities, and not shown in the BPMN model. \
        You have to infer them basing on the activities. For example, if an activity is "Submit Request," the process state could be "Request Submitted."
        - Provide a clear and concise mapping of activities to their corresponding process states.
        
        Return your analysis in the following format:
        - activity: <activity_name>
        process_state: <process_state_name>

        List each activity and its corresponding process state separately using the format above.
        """

        prompt = PromptTemplate.from_template(sys_prompt)
        
        # Use the Pydantic model directly, not as a dictionary
        structured_llm = llm.with_structured_output(ListActivityProcessState)

        # build the chain with the prompt and the structured LLM
        chain = prompt | structured_llm
        
        ## Invoke the LLM with the model string
        act_states = chain.invoke({'model': self.model_string})

        return act_states

    def extract_all_process_data(self)-> dict:
        """Extracting all necessary information from the BPMN model, which are needed for the simulation."""
        try:
            process_summary = self.extract_process_summary()
            role_actions = self.extract_role_actions()
            start_end_events = self.extract_start_end_events()
            act_states = self.extract_act_states()

            return {
                "process_summary": process_summary.process_sumary,
                "role_actions": {role_action.role: role_action.actions for role_action in role_actions.role_action_list},
                "start_end_events": {
                    "start_events": start_end_events.start_events,
                    "first_activities": start_end_events.first_activities,
                    "end_events": start_end_events.end_events,
                    "last_activities": start_end_events.last_activities
                },
                "act_states": {act_state.activity: act_state.process_state for act_state in act_states.activity_process_states}
            }
        except Exception as e:
            raise RuntimeError(f"An error occurred during BPMN extraction: {e}")
