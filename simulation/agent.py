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


class Demographics(BaseModel):
    """Demographics of the agent"""
    name: str = Field(description="Name of the agent")
    age: str = Field(description="Age of the agent")
    gender: str = Field(description="Gener of the agent")
class AgentOutput(BaseModel):
    """Structured Agent Output"""
    process_id: str = Field(description="Unique identifier for the process")
    action: str = Field(description="Action taken by the agent")
    timestamp: str = Field(description="Timestamp of the action")

class ListAgentOutput(BaseModel):
    """Structured Agent Output List"""
    agent_outputs: List[AgentOutput] = Field(description="List of agent outputs")


class Agent(BaseModel):
    role: str = Field(description="Agent's role in the process")
    demographics_attributes: List[str] = Field(..., description="Agent's demographical attribute such as name, age, gender, etc.")
    demographics_values: List[str] = Field(default_factory=list, description="Agent's demographical values such as John (for name), 25 (for age), etc.")
    actions: List[str] = Field(..., description="Agent's possible actions")
    prompt: str = Field(default='', description="Prompt for the agent")
    available: bool = Field(default=True, description="Agent's availability status")
    
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
        print(f'random_text: {random_text}\n')
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
        structured_llm = llm.with_structured_output(demographics_model)
        chain = prompt | structured_llm
        demographics = chain.invoke({"role": self.role, "demographics_list": demographics_list, "random_text": random_text})
        print(f'demographics: {demographics}\n')
        
        # write to demographics values
        self.demographics_values = [demographics.model_dump()[attr] for attr in self.demographics_attributes]

        return demographics
    
    # Add a function to evaluate the generated demographics and re-generate new ones if needed

    def create_prompt(self, llm = llm_high):
        """Create prompt for the agent"""
        role = self.role
        actions = ', '.join(self.actions)
        # get demographics values of the agent
        agent_demographics = self._generate_demographics(llm=llm)
        agent_demographics = ' '.join([f"{key}: {value}" for key, value in agent_demographics.model_dump().items()])
        prompt = f"""
        You are an agent participating in a business process.
        You are involved in the following process: {{process_summary}}
        You play the role of: {role}.
        Your demographic profile is: {agent_demographics}.
        You have the ability to perform the following actions: {actions}
        The current state of the process is: {{process_state}}.
        The current timestamp is: {{current_time}}.
        The process may be influenced by the following external factors: {{external_factors}}
        The current external conditions are: {{external_values}}.
        Based on the above information, determine whether to take an action. If you choose to act, specify the action and record the timestamp.

        Please respond using the following structured format:
        - process_id: ```<process_id>```
        - action: ```<your_chosen_action>```
        - timestamp: ```<timestamp_of_action>```

        If no action is taken, leave all fields empty.
        You may take multiple actions. If so, repeat the format above for each action taken.

        Pay attention to the current time, the current process state, the actions you're allowed to take, the process workflow, and the external factors when making your decision.
        """
        self.prompt = prompt
    
    def take_action(self, process_summary:str, process_state:str, current_time:str, external_factors:list, external_values:dict, llm=llm_mid) -> AgentOutput:
        """Take action based on the prompt and return the output"""

        structured_llm = llm.with_structured_output(ListAgentOutput)
        prompt = PromptTemplate.from_template(self.prompt)
        chain = prompt | structured_llm
        agent_output = chain.invoke({
            "process_summary": process_summary,
            "process_state": process_state,
            "current_time": current_time,
            "external_factors": external_factors,
            "external_values": external_values
        })
        return agent_output