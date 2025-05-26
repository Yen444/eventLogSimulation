from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate

from .agent_manager import Agent, AgentManager
from .config_manager import ConfigManager
from .process_manager import ProcessManager, logEvent

from collections import OrderedDict
import datetime
import os

from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)


class ExternalFactorsManager:
    def __init__(self, config_manager: ConfigManager, llm=llm):
        self.config_manager = config_manager
        self.llm = llm
        self.recent_external_values = OrderedDict()

    def _generate_external_factors_model(self):
        """Generate external factors Pydantic model."""
        if self.config_manager.configs is None:
            simulation_configs = self.config_manager.update_and_load_configs()
        else:
            simulation_configs = self.config_manager.configs

        external_factors = simulation_configs.get('external_factors', [])
        if not external_factors:
            raise ValueError("External factors are required.")
        fields = {field: (str, Field(..., description=f"{field.capitalize()} condition")) for field in external_factors}
        return create_model("ExternalFactors", **fields)
    
    def _get_recent_external_values(self)->str:
        """"Get recent external values in string"""
        recent_external_values = ""
        for key, value in self.recent_external_values.items():
            value_str = ', '.join([f"{k}: {v}" for k, v in value.items()])
            recent_external_value = f'External conditions at {key} is: {value_str}.'
            recent_external_values += recent_external_value
        return recent_external_values
    
    def generate_external_values(self, current_time)->str:
        """Generate external values for the current step."""
        if self.config_manager.configs is None:
            simulation_configs = self.config_manager.update_and_load_configs()
        else:
            simulation_configs = self.config_manager.configs
        external_factors = simulation_configs.get('external_factors', [])

        external_factors_model = self._generate_external_factors_model()

        sys_prompt = """
        You are simulating a business process influenced by external contextual factors.
        Given the following external factors: {external_factors}, generate their current values based on the current time: {current_time} and past values: {past_external_values}.
        The output should be in a structured format, listing each external factor and its corresponding value:
        - <external_factor>: <value>
        - <external_factor>: <value>
        ...
        """
        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = self.llm.with_structured_output(external_factors_model)
        chain = prompt | structured_llm
        external_values = chain.invoke({
            "external_factors": ', '.join(external_factors),
            "current_time": current_time,
            "past_external_values": self.recent_external_values
        })
        # Only keep at most 10 recent external values
        if len(self.recent_external_values) >= 10:
            self.recent_external_values.popitem(last=False)
        # add new external values to the recent external values    
        self.recent_external_values[current_time] = external_values.model_dump()

        return ', '.join([f"{k}: {v}" for k, v in external_values.model_dump().items()])


class SimulationEngine:
    def __init__(self, bpmn_path, extracted_config_path, user_config_path, simulation_config_path, log_path):
        self.config_manager = ConfigManager(
            bpmn_path=bpmn_path,
            extracted_config_path=extracted_config_path,
            user_config_path=user_config_path,
            simulation_config_path=simulation_config_path
        )
        self.agent_manager = AgentManager(self.config_manager)
        self.process_manager = ProcessManager(log_path)
        self.external_factors_manager = ExternalFactorsManager(self.config_manager)

    def step(self, agent: Agent, current_time):
        """Run a single step of the simulation."""

        process_summary = self.config_manager.configs.get('process_summary', '')
        process_state = self.process_manager.get_incomplete_process_state()
        external_factors = self.config_manager.configs.get('external_factors', [])
        external_factors = ', '.join(external_factors)
        external_values = self.external_factors_manager.generate_external_values(current_time)

        # agent takes action
        if agent.trigger:
            print(f'taking trigger action')
            out = agent.take_trigger_action(
                process_summary=process_summary,
                current_time=current_time,
                external_factors=external_factors,
                external_values=external_values
            )
            print(f'out: {out}')
            for agent_output in out.agent_outputs:
                if agent_output.action != None:
                    log_event = logEvent(
                        process_id=self.process_manager.generate_process_id(),
                        action=agent_output.action,
                        timestamp=agent_output.timestamp,
                        external_values=external_values,
                        internal_values=json.dumps({atr:getattr(agent_output.data, atr) for atr in agent_output.data.model_fields.keys()}),
                        #feedback=agent_output.feedback,
                        agent=agent
                    )
                    self.process_manager.update_process(log_event=log_event, last_activities=self.config_manager.configs.get('start_end_events', {}).get('last_activities', []))

        else:
            out = agent.take_action(
                process_summary=process_summary,
                process_state=process_state,
                current_time=current_time,
                external_factors=external_factors,
                external_values=external_values
            )
            for agent_output in out.agent_outputs:
                if agent_output.action != None:
                    log_event = logEvent(
                        process_id= agent_output.process_id,
                        action=agent_output.action,
                        timestamp=agent_output.timestamp,
                        external_values=external_values,
                        internal_values=json.dumps({atr:getattr(agent_output.data, atr) for atr in agent_output.data.model_fields.keys()}),
                        #feedback=agent_output.feedback,
                        agent=agent
                    )
                    self.process_manager.update_process(log_event=log_event, last_activities=self.config_manager.configs.get('start_end_events', {}).get('last_activities', []))
        
    def run(self, start_time, end_time, inter_arrival_time):
        """Run the simulation."""
        print(f'creating cordinator agent')
        self.agent_manager.create_cordinator_agent()
        print(f'creating organization agents')
        self.agent_manager.create_org_agents()
        start_time = datetime.datetime.fromisoformat(start_time)
        end_time = datetime.datetime.fromisoformat(end_time)
        inter_arrival_time = datetime.timedelta(seconds=inter_arrival_time)

        current_time = start_time
        while current_time < end_time:
            print(f"Current time: {current_time}")

            # We have to initialize the process by creating and calling the trigger agent
            print(f'Creating trigger agent on the fly')
            trigger_agent = self.agent_manager.create_trigger_agent_onfly()
            #print(f'Trigger agent prompt: {trigger_agent.prompt}')
            # trigger agent takes action
            print(f'Trigger agent takes action')

            self.step(trigger_agent, current_time)

            # cordinator agent takes action
            print(f'Cordinator hands off the process to the organization agents')
            process_states = self.process_manager.get_incomplete_process_state()
            org_agents = self.agent_manager.org_agents
            external_agents = self.agent_manager.ext_agents
            cordinator_outputs = self.agent_manager.cordinator.hand_off(process_states=process_states, org_agents=org_agents, external_agents=external_agents)
            print(f'Cordinator outputs: {cordinator_outputs}')

            for cordinator_output in cordinator_outputs.cordinator_outputs:
                print(type(cordinator_output))
                process_id = cordinator_output.process_id
                agent_id = cordinator_output.agent_id
                agent_role = cordinator_output.agent_role
                if agent_id:
                    agent = self.agent_manager.get_agent_by_id(agent_id)
                elif agent_role:
                    agent = self.agent_manager.create_ext_agent_onfly(agent_role)
                else:
                    raise ValueError("Either agent_id or agent_role must be provided.")
                # agent takes action
                print(f'Agent {agent.id} {agent.role} takes action')
                self.step(agent, current_time)

            