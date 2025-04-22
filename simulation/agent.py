from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)


class AgentOutput(BaseModel):
    """Structured Agent Output"""
    action: str = Field(description="Action taken by the agent")
    timestamp: str = Field(description="Timestamp of the action")


class Agent(BaseModel):
    process_summary: str = Field(description="Summary of the process")
    role: str = Field(description="Agent's role in the process")
    demographics_attributes: List[str] = Field(description="Agent's demographical attribute such as name, age, gender, etc.")
    demographics_values: List[str] = Field(description="Agent's demographical values such as John (for name), 25 (for age), etc.")
    external_factors: List[str] = Field(description="External factors which can affect the agent's action and the whole process")
    actions: List[str] = Field(description="Agent's possible actions")

    def _generating_demographics_model(self):
        """Generate dynamaic demographics pydantic model"""
        fields = {field: (str, Field(None, description=f"{field.capitalize()} of the agent")) for field in self.demographics_attributes}
        demographics_model = create_model("Demographics", **fields)
        return demographics_model
    
    def _generate_demographics(self, llm = llm):
        """Generate demographics of the agent"""
        if not self.role and self.demographics_attributes:
            raise ValueError("Role and demographics are required to generate demographics.")
        # create string from list of demographics
        demographics_list = ', '.join(self.demographics_attributes)
        # pydantic model for demographics
        demographics_model = self._generating_demographics_model()

        sys_prompt = """Given a person with the role of a {self.role}, randomly generate demographics of that person. \
        The demographics should include the following attributes: {demographics_list}.
        Return the output in a structured format, listing each attribute and its corresponding value:
        - attribute: value
        - attribute: value
        ...
        """
        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = llm.with_structured_output(demographics_model)
        chain = prompt | structured_llm
        demographics = chain.invoke({"role": self.role, "demographics_list": demographics_list})
        # write to demographics values
        self.demographics_values = [demographics.attribute for acttribute in self.demographics.attributes]

        return demographics

    def create_prompt(self, llm = llm, process_state:str = None):
        """Create prompt for the agent"""
        # get demographics values of the agent
        agent_demographics = self._generate_demographics(llm=llm)
        agent_demographics_dict = agent_demographics.model_dump()

        prompt = """You are an agent in a business process.
        You take part in the following process: {self.process_summary}
        You are a {self.role} in the process.
        Your demographics are: {agent_demographics_dict}
        The external factors which can affect your action and the whole process are: {self.external_factors}.
        You can take the following actions: {self.actions}.
        The current state of the process is: {process_state}.
        Based on the above information, take an action and record the timestamp of the action.
        Output your action and the timestamp in a structured format:
        - action: ```action```
        - timestamp: ```timestamp```
        """
        return prompt
    
    def take_action(self, llm = llm, process_state:str = None) -> AgentOutput:
        """Take action based on the prompt and return the output"""
        sys_prompt = self.create_prompt(llm=llm, process_state=process_state)
        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = llm.with_structured_output(AgentOutput)
        chain = prompt | structured_llm
        agent_output = chain.invoke({"process_state": process_state})
        return agent_output