from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate


from .llm_utils import generate_random_text_with_mem_chatbot
from .utils import generate_agent_id
from .config_manager import ConfigManager


from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
api_key_openai = os.getenv('OPENAI_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

llm_low = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0) # gemini-2.5-pro-preview-03-25
llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5)
llm_high = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)
llm_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    openai_api_key=api_key_openai
)

class VerifyOutput(BaseModel):
    appropriate: bool = Field(description="Whether the action is appropriate or not")

class InternalDataModel(BaseModel):
    vehicle: Optional[str] = Field(default=None, description="Car model name")
    registration_number: Optional[str] = Field(default=None, description="Registration number of the car")
    incident_description: Optional[str] = Field(default=None, description="Incident description")
    damage: Optional[str] = Field(default=None, description="Damage of the car")
    payment_amount: Optional[str] = Field(default=None, description="Payment amount of the repairment")

class FeedbackModel(BaseModel):
    task_difficulty: Optional[float] = Field(default=None, description="Difficulty of the task")
    satisfaction_score: Optional[float] = Field(default=None, description="Satisfaction score when doing the task")
    time_taken: Optional[float] = Field(default=None, description="Time taken to complete the task in hours")

######## Output Model for Trigger Agent ########
class TrigerOutput(BaseModel):
    """Structured Trigger Output"""
    action: str = Field(description="Action taken by the agent")
    timestamp: str = Field(description="Timestamp of the action")
    data: InternalDataModel
    feedback: Optional[FeedbackModel] = Field(default=None, description="Feedback from the agent about the action they took")

class ListTrigerOutput(BaseModel):
    """Structured Trigger Output List"""
    agent_outputs: List[TrigerOutput] = Field(description="List of agent outputs")

######## Output Model for normal Agent ########

class AgentOutput(BaseModel):
    process_id: Optional[str] = Field(description="Unique identifier for the process")
    action: str = Field(description="Action taken by the agent")
    timestamp: str = Field(description="Timestamp of the action")
    data: InternalDataModel
    feedback: Optional[FeedbackModel] = Field(default=None, description="Feedback from the agent about the action they took")

class ListAgentOutput(BaseModel):
    """Structured Agent Output List"""
    agent_outputs: List[AgentOutput] = Field(description="List of agent outputs")

class Agent(BaseModel):
    id: str = Field(description="Unique identifier for the agent")
    role: str = Field(description="Agent's role in the process")
    demographics_attributes: List[str] = Field(..., description="Agent's demographical attribute such as name, age, gender, etc.")
    demographics_values: List[str] = Field(default_factory=list, description="Agent's demographical values such as John (for name), 25 (for age), etc.")
    actions: List[str] = Field(..., description="Agent's possible actions")
    action2data: Dict[str, List[str]] = Field(default_factory=dict, description="Mapping of actions to internal data attributes")
    prompt: str = Field(default='', description="Prompt for the agent")
    verification_prompt: str = Field(default='', description="Verification prompt for the agent")
    trigger: bool = Field(default=False, description="Trigger agent or not")
    #memory: Optional[Any] = Field(default=None, description="Memory of the agent, used for storing previous actions and decisions")
    
    # Create a dynamic pydantic model for demographics
    def _generating_demographics_model(self):
        """Generate dynamaic demographics pydantic model"""
        fields = {field: (str, Field(..., description=f"{field.capitalize()} of the agent")) for field in self.demographics_attributes}
        demographics_model = create_model("Demographics", **fields)
        return demographics_model
    
    # Generate demographics of the agent
    def _generate_demographics(self, llm):
        """Generate demographics of the agent"""
        if not self.role or not self.demographics_attributes:
            raise ValueError("Role and demographics are required to generate demographics.")
        # create string from list of demographics
        demographics_list = ', '.join(self.demographics_attributes)
        # pydantic model for demographics
        demographics_model = self._generating_demographics_model()
        # add random text to the prompt
        random_text = generate_random_text_with_mem_chatbot()
        #print(f'random_text: {random_text}\n')
        # create prompt for demographics
        sys_prompt = """
        Given a person with the role of a {role}, randomly generate demographics of that person. \
        The demographics should include the following attributes: {demographics_list}.
        {random_text}
        Return the output in a structured format, listing each attribute and its corresponding value:
        - <attribute>: <value>
        - <attribute>: <value>
        - ...
        """
        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = llm_openai.with_structured_output(demographics_model)
        chain = prompt | structured_llm
        demographics = chain.invoke({"role": self.role, "demographics_list": demographics_list, "random_text": random_text})
        #print(f'demographics: {demographics}\n')
        
        # write to demographics values
        self.demographics_values = [demographics.model_dump()[attr] for attr in self.demographics_attributes]

        return demographics
    

    def create_prompt(self, llm = llm_high):
        """Create prompt for the agent"""
        role = self.role
        actions = ', '.join(self.actions)
        # get demographics values of the agent
        agent_demographics = self._generate_demographics(llm=llm)
        agent_demographics = ' '.join([f"{key}: {value}" for key, value in agent_demographics.model_dump().items()])
        # convert action2data to string
        action2data_str = ', '.join(f"{action}: {', '.join(factors)}" for action, factors in self.action2data.items())

        prompt = f"""
        You are an agent participating in a business process.

        Process Summary: {{process_summary}}  
        Your Role: {role}  
        Demographic Profile: {agent_demographics}  
        Available Actions: {actions}  

        Each action may require specific data. Below is a mapping of actions to their required data fields:  
        Action Data Mapping: {action2data_str} 

        You are assigned to the following current process instance:
        Current Process State: {{process_state}}  

        You are given the following information:
        Current Timestamp: {{current_time}}  
        Relevant External Factors: {{external_factors}}  
        Current External Conditions: {{external_values}}  

        Based on the above information, decide whether you will finish any action for the assigned process within the next hour.  
        If you will finished the action, specify the process_id, the action(s), the timestamp(s) at which you finished your action, and include your feedback (difficulty_score from 1 to 5, satisfaction score from 1 to 5, and the time_taken to finish the action in hours). The timestamp(s) must be within one hour from the current time.

        **Please use the following structured format for each action taken:**

        - process_id: ```<process_id>```  
        - action: ```<your_chosen_action>```  
        - timestamp: ```<timestamp_of_action>```  
        - data: ```<data_attribute_1>: <data_value_1>, <data_attribute_2>: <data_value_2>, ...```  
        - satisfaction_score: ```<satisfaction_score from 1 to 5>```
        - difficulty_score: ```<difficulty_score from 1 to 5>```
        - time_taken: ```<time_taken in hours>``` 

        **If you think you can't finish any action in the next one hour:**
        - Leave the `action` field and other fields empty.

        **If an action does not require any data:**
        - Leave the `data` field empty.

        **Only output the data needed for the action you will take, which is specified in the action data mapping above.**

        **Output a suitable timestamp for the action you will finish. The timestamp must satisfy all of the following conditions:**
        - It must be within one hour from the current time.
        - It must not be earlier than the current time.
        - It must not be earlier than the timestamp of the ``SUBMIT CLAIM`` action. Because this action is the first action in the process and it must be finished before any other action.

        **If you finish multiple actions:**
        - Repeat the above structured format for each action.

        **You can take consecutive actions, but you must ensure that the timestamp of each action is within one hour from the current time.**
        - For example, if you are done with `perform assessment`, and you know that you reject the claim, you can also finish the action `reject claim` right after that.
        """

        self.prompt = prompt
    
    def create_verification_prompt(self, llm = llm_high):
        """Create verification prompt for the agent"""
        role = self.role
        actions = ', '.join(self.actions)
        # get demographics values of the agent
        agent_demographics = self._generate_demographics(llm=llm)
        agent_demographics = ' '.join([f"{key}: {value}" for key, value in agent_demographics.model_dump().items()])
        # convert action2data to string
        action2data_str = ', '.join(f"{action}: {', '.join(factors)}" for action, factors in self.action2data.items())
        verification_prompt = f"""
        You are a reflective agent reviewing your previously decided actions in a business process.

        Here is a summary of the situation:
        - Process Summary: {{process_summary}}
        - Your Role: {role}
        - Demographic Profile: {agent_demographics}
        - Available Actions: {actions}

        Contextual Information:
        - Current Timestamp: {{current_time}}
        - Current Process State: {{process_state}}
        - Relevant External Factors: {{external_factors}}
        - Current External Conditions: {{external_values}}

        You did plan to finish the following action:
        {{previous_action}}

        **Your Task:**
        Evaluate the *appropriateness* of the previously proposed action based *primarily* on the timestamp associated with that action. For the action, assess the following:

        1. **Is the timestamp appropriate:**  
        - Does the timestamp fall within a typical and reasonable time window for work-related activities?  
        - Avoid suggesting actions during biologically or culturally inappropriate times, such as deep night hours (e.g., 22:00-06:00), unless explicitly justified by the context.

        2. **Personal Feasibility:**  
        - Are you, given your role and demographic profile, realistically able to complete the action at the specified timestamp?  
        - Consider your energy levels, time availability, and workload distribution. If the timestamp is likely to hinder your ability to complete the action effectively, mark it as inappropriate.

        **For each action, determine whether its timestamp is both appropriate and personally feasible:**

        - If the action meets both criteria, output True.

        - If it fails either criterion, output False.

        **Output format:**
        - appropriate: ```<True or False>```

        """
        self.verification_prompt = verification_prompt

    
    def create_trigger_prompt(self, llm = llm_high):

        """Create prompt for the trigger agent"""
        role = self.role
        actions = ', '.join(self.actions)
        # get demographics values of the agent
        agent_demographics = self._generate_demographics(llm=llm)
        agent_demographics = ' '.join([f"{key}: {value}" for key, value in agent_demographics.model_dump().items()])
        action2data_str = ', '.join(f"{action}: {', '.join(factors)}" for action, factors in self.action2data.items())
        prompt = f"""
        You are an agent participating in a business process.

        Process Summary: {{process_summary}}  
        Your Role: {role}  
        Demographic Profile: {agent_demographics}  
        Available Actions: {actions}  

        Each action may require specific data. Below is a mapping of actions to their required data fields:  
        Action Data Mapping: {action2data_str} 

        You are given the following information
  
        Current Timestamp: {{current_time}}  
        Relevant External Factors: {{external_factors}}  
        Current External Conditions: {{external_values}}  

        Based on the above information, decide whether to take one or more actions within the next hour.  
        If you choose to act, specify the action(s), the timestamp(s), and include your feedback (a satisfaction score from 1 to 5). The timestamp(s) must be within one hour of the current time.

        **Please use the following structured format for each action taken:**
 
        - action: ```<your_chosen_action>```  
        - timestamp: ```<timestamp_of_action>```  
        - data: ```<data_attribute_1>: <data_value_1>, <data_attribute_2>: <data_value_2>, ...```  
        - feedback: ```<satisfaction_score from 1 to 5, difficulty_score from 1 to 5, time_taken in hours>```   

        **If you don't want to take any action:**
        - Leave the `action` field and other fields empty.

        **If an action does not require any data:**
        - Leave the `data` field empty.
    
        **Only output the data needed for the action you will take, which is specified in the action data mapping above.**

        **If you take multiple actions:**
        - Repeat the above structured format for each action.
        """
        self.prompt = prompt
    
    def create_trigger_verification_prompt(self, llm = llm_high):
        """Create verification prompt for the trigger agent"""
        role = self.role
        actions = ', '.join(self.actions)
        # get demographics values of the agent
        agent_demographics = self._generate_demographics(llm=llm)
        agent_demographics = ' '.join([f"{key}: {value}" for key, value in agent_demographics.model_dump().items()])
        action2data_str = ', '.join(f"{action}: {', '.join(factors)}" for action, factors in self.action2data.items())
        verification_prompt = f"""
        You are a reflective agent reviewing the actions you previously proposed in a business process.

        Here is a summary of the situation:
        - Process Summary: {{process_summary}}  
        - Your Role: {role}  
        - Demographic Profile: {agent_demographics}  
        - Available Actions: {actions}  
        - Action Data Mapping: {action2data_str}  

        Contextual Information:
        - Current Timestamp: {{current_time}}
        - Relevant External Factors: {{external_factors}}
        - Current External Conditions: {{external_values}}

        You did plan to take the following action:
        {{previous_action}}

        **Your Task:**
        Evaluate the *appropriateness* of each previously proposed action based *primarily* on the timestamp associated with that action. For each action, assess the following:

        1. **Is the timestamp appropriate:**  
        - Does the timestamp fall within a typical and reasonable time window for work-related activities?  
        - Avoid suggesting actions during biologically or culturally inappropriate times, such as deep night hours (e.g., 22:00-06:00), unless explicitly justified by the context.

        2. **Personal Feasibility:**  
        - Are you, given your role and demographic profile, realistically able to complete the action at the specified timestamp?  
        - Consider your energy levels, time availability, and workload distribution. If the timestamp is likely to hinder your ability to complete the action effectively, mark it as inappropriate.

        **For each action, determine whether its timestamp is both appropriate and personally feasible:**

        - If the action meets both criteria, output True.

        - If it fails either criterion, output False.

        **Output format:**
        - appropriate: ```<True or False>```

        """
        self.verification_prompt = verification_prompt

    
    def take_trigger_action(self, process_summary:str, current_time:str, external_factors:list, external_values:dict, llm=llm_mid) -> AgentOutput:
        """Take action based on the prompt and return the output"""
        structured_llm = llm.with_structured_output(ListTrigerOutput)
        prompt = PromptTemplate.from_template(self.prompt)
        chain = prompt | structured_llm
        #chain = prompt | llm
        agent_output = chain.invoke({
            "process_summary": process_summary,
            "current_time": current_time,
            "external_factors": external_factors,
            "external_values": external_values
        })
        return agent_output
    
    def take_action(self, process_summary:str, process_state:str, current_time:str, external_factors:list, external_values:dict, llm=llm_mid) -> AgentOutput:
        """Take action based on the prompt and return the output"""

        structured_llm = llm.with_structured_output(ListAgentOutput)
        prompt = PromptTemplate.from_template(self.prompt)
        chain = prompt | structured_llm
        #chain = prompt | llm
        agent_output = chain.invoke({
            "process_summary": process_summary,
            "process_state": process_state,
            "current_time": current_time,
            "external_factors": external_factors,
            "external_values": external_values
        })
        return agent_output
    def take_action_verify(self, process_summary:str, process_state:str, current_time:str, external_factors:list, external_values:dict, llm=llm_mid) -> AgentOutput:
        first_actions = self.take_action(
            process_summary=process_summary,
            process_state=process_state,
            current_time=current_time,
            external_factors=external_factors,
            external_values=external_values)

        structured_llm = llm_openai.with_structured_output(VerifyOutput)
        prompt = PromptTemplate.from_template(self.verification_prompt)
        chain = prompt | structured_llm
        to_remove = []
        for action in first_actions.agent_outputs:
            action_string = (
                f'Process_id: {action.process_id}, '
                f'action: {action.action}, '
                f'timestamp: {action.timestamp}, '
                f'data: {", ".join([f"{key}: {value}" for key, value in (action.data.model_dump().items() if action.data else []) if value is not None])}, '
                f'feedback: {", ".join([f"{key}: {value}" for key, value in (action.feedback.model_dump().items() if action.feedback else []) if value is not None])}\n'
            )
            verified_agent_output = chain.invoke({
                "process_summary": process_summary,
                "process_state": process_state,
                "current_time": current_time,
                "external_factors": external_factors,
                "external_values": external_values,
                "previous_action": action_string
            })
            if verified_agent_output.appropriate == False:
                # remove action from the first actions
                print(f'ACTION IS REMOVED')
                to_remove.append(action)

        first_actions.agent_outputs = [action for action in first_actions.agent_outputs if action not in to_remove]
        return first_actions

    def take_trigger_action_verify(self, process_summary:str, current_time:str, external_factors:list, external_values:dict, llm=llm_mid) -> AgentOutput:
        first_actions = self.take_trigger_action(
            process_summary=process_summary,
            current_time=current_time,
            external_factors=external_factors,
            external_values=external_values)

        print(f'Trigger agent first action: {first_actions}')

        structured_llm = llm_openai.with_structured_output(VerifyOutput)
        prompt = PromptTemplate.from_template(self.verification_prompt)
        chain = prompt | structured_llm
        to_remove = []
        for action in first_actions.agent_outputs:
            action_string = (
                f'action: {action.action}, '
                f'timestamp: {action.timestamp}, '
                f'data: {", ".join([f"{key}: {value}" for key, value in (action.data.model_dump().items() if action.data else []) if value is not None])}, '
                f'feedback: {", ".join([f"{key}: {value}" for key, value in (action.feedback.model_dump().items() if action.feedback else []) if value is not None])}\n'
            )

            verified_agent_output = chain.invoke({
                "process_summary": process_summary,
                "current_time": current_time,
                "external_factors": external_factors,
                "external_values": external_values,
                "previous_action": action_string
            })
            if verified_agent_output.appropriate == False:
                # remove action from the first actions
                print(f'ACTION IS REMOVED')
                to_remove.append(action)

        first_actions.agent_outputs = [action for action in first_actions.agent_outputs if action not in to_remove]
        
        return first_actions


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

    def hand_off(self, process_states, org_agents, external_agents):
        org_agents_str = ', '.join([f"ID: {agent.id} (role: {agent.role}, actions: {agent.actions})" for agent in org_agents])
        external_agents_str = ', '.join([f"ID: {agent.id} (role: {agent.role}, actions: {agent.actions})" for agent in external_agents])
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

        structured_llm = llm_high.with_structured_output(ListCordinatorOutput)
        prompt = PromptTemplate.from_template(prompt)
        chain = prompt | structured_llm
        cordinator_output = chain.invoke({
            "process_summary": self.process_summary,
            "process_states": process_states,
            "org_agents": org_agents_str,   
            "external_agents": external_agents_str
        })
        return cordinator_output


class AgentManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.org_agents: List[Agent] = []
        self.ext_agents: List[Agent] = []
        self.cordinator: Cordinator = None
    
    
    def create_trigger_agent_onfly(self):
        """Create a trigger agent based on the configuration."""
        simulation_configs = self.config_manager.update_and_load_configs()
        role_actions = simulation_configs.get('trigger_role_actions', {})
        demographics = simulation_configs.get('demographics', [])
        trigger_agent = next(iter(role_actions)) if isinstance(role_actions, dict) else role_actions
        action2data = simulation_configs.get('action_data', {}) 

        if trigger_agent:
            used_ids = [agent.id for agent in self.org_agents] + [agent.id for agent in self.ext_agents]
            agent_id = generate_agent_id(used_ids=used_ids)
            agent = Agent(
                id=agent_id,
                role=trigger_agent,
                demographics_attributes=demographics,
                actions=role_actions.get(trigger_agent, []),
                action2data=action2data,
                trigger=True
            )
            agent.create_trigger_prompt()
            agent.create_trigger_verification_prompt()
            self.ext_agents.append(agent)
            return agent
        else:
            raise ValueError("Trigger agent role is not defined in the configuration.")


    def create_org_agents(self):
        """Create agents based on the configuration."""
        simulation_configs = self.config_manager.update_and_load_configs()
        role_actions = simulation_configs.get('org_role_actions', {})
        demographics = simulation_configs.get('demographics', [])
        agent_numbers = simulation_configs.get('num_org_agents', {})
        action2data = simulation_configs.get('action_data', {})

        for agent_role, actions in role_actions.items():
            num_agents = agent_numbers.get(agent_role, 2)
            for _ in range(num_agents):
                used_ids = [agent.id for agent in self.org_agents] + [agent.id for agent in self.ext_agents]
                agent_id = generate_agent_id(used_ids=used_ids)
                agent = Agent(
                    id=agent_id,
                    role=agent_role,
                    demographics_attributes=demographics,
                    actions=actions,
                    action2data=action2data
                )
                agent.create_prompt()
                agent.create_verification_prompt()
                self.org_agents.append(agent)

    def create_ext_agent_onfly(self, role:str):
        """Create agents on the fly based on the cofigurations or handoff"""
        simulation_configs = self.config_manager.update_and_load_configs()
        role_actions = simulation_configs.get('ext_role_actions', {})
        demographics = simulation_configs.get('demographics', [])
        action2data = simulation_configs.get('action_data', {})
        
        used_ids = [agent.id for agent in self.org_agents] + [agent.id for agent in self.ext_agents]
        agent_id = generate_agent_id(used_ids=used_ids)
        agent = Agent(
            id=agent_id,
            role=role,
            demographics_attributes=demographics,
            actions=role_actions.get(role, []),
            action2data=action2data
        )
        agent.create_prompt()
        agent.create_verification_prompt()
        self.ext_agents.append(agent)
        return agent
    
    def create_cordinator_agent(self):
        """Create a cordinator agent"""
        simulation_configs = self.config_manager.update_and_load_configs()
        process_summary = simulation_configs.get('process_summary', '')
        cordinator = Cordinator(
            role='cordinator',
            process_summary=process_summary
        )
        self.cordinator = cordinator 
    
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by its ID."""
        for agent in self.org_agents + self.ext_agents:
            if agent.id == agent_id:
                return agent

