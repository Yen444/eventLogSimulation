from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate

from simulation.utils import read_bpmn_file

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)

############################
# Extracting BPMN Roles and Actions
############################

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

def extract_role_actions(bpmn_path, llm=llm):
    """This function reads a BPMN diagram (in XML format), prompts an LLM to analyze it,
    and returns structured output that maps each identified role to a list of actions
    they perform in the process. The extracted information can be used to define agents
    and behaviors for simulation or process analysis.

    Args:
        bpmn_path (str): Path to the BPMN file (.bpmn).
        llm: An LLM capable of structured output (https://python.langchain.com/docs/integrations/chat/) 
    used to analyze the BPMN model.

    Returns:
        ListRoleAction: A Pydantic object containing a list of RoleAction objects, 
        where each object represents a process participant (role) and their corresponding actions.
    """
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

    chain = prompt | structured_llm

    # Read the BPMN model from the file
    model_string = read_bpmn_file(bpmn_path=bpmn_path)
    
    ## Invoke the LLM with the model string
    role_act_list = chain.invoke({'model': model_string})
    return role_act_list
    

############################
# Extracting start/ end events and activities
############################

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

def extract_start_end_events(bpmn_path, llm=llm):
    """This function reads a BPMN diagram (in XML format), prompts an LLM to analyze it,
    and returns structured output of start events, their following first activities, end events, and their preceding last activities.

    Args:
        file_path (str): Path to the BPMN file (.bpmn).
        llm: An LLM capable of structured output (https://python.langchain.com/docs/integrations/chat/) 
    used to analyze the BPMN model.

    Returns:
        StartEndEvents: A Pydantic object containing a list of start events, first activities, end events, and last activities. 
    """
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

    chain = prompt | structured_llm

    # Read the BPMN model from the file

    model_string = read_bpmn_file(bpmn_path=bpmn_path)
    
    ## Invoke the LLM with the model string
    start_end_events = chain.invoke({'model': model_string})

    return start_end_events   

############################
# Activities and Process States Mapping
############################

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

def extract_act_states(bpmn_path, llm=llm):
    """This function reads a BPMN diagram (in XML format), prompts an LLM to analyze it,
    and returns structured output that maps each identified activity to its corresponding process state.

    Args:
        bpmn_path (str): Path to the BPMN file (.bpmn).
        llm: An LLM capable of structured output (https://python.langchain.com/docs/integrations/chat/) 
    used to analyze the BPMN model.

    Returns:
        ListActivityProcessState: A Pydantic object containing a list of ActivityProcessState objects,
        where each object represents a BPMN activity and its corresponding process state. 
    """

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

    chain = prompt | structured_llm

  
    model_string = read_bpmn_file(bpmn_path=bpmn_path)
    
    ## Invoke the LLM with the model string
    act_states = chain.invoke({'model': model_string})

    return act_states


############################
# Mapping all extracting results
############################

def bpmn_extraction(bpmn_path, llm):
    """This function combines the above three functions to extract roles and their corresponding actions, start/end events, first/last activities, 
    as well as mapping between an activity and the process state from a BPMN diagram.

    Args:
        bpmn_path (str): Path to the BPMN file (.bpmn).
        llm: An LLM capable of structured output (https://python.langchain.com/docs/integrations/chat/) 
        used to analyze the BPMN model.

    Returns:
        dict:
    """
    try:
        role_actions = extract_role_actions(bpmn_path=bpmn_path, llm=llm)
        start_end_events = extract_start_end_events(bpmn_path=bpmn_path, llm=llm)
        act_states = extract_act_states(bpmn_path=bpmn_path, llm=llm)

        return {
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