from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any, Type
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import HumanMessage

from .utils import extract_time_parts
from .config_manager import ConfigManager
from .prompt_manager import PromptManager
from .tool_manager import ToolManager

from dotenv import load_dotenv
import os

from deepdiff import DeepDiff
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
api_key_openai = os.getenv('OPENAI_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

llm_low = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0) # gemini-2.5-pro-preview-03-25
llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5)
llm_high = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)

llm_openai = ChatOpenAI(
    model="gpt-5-nano",
    temperature=1.0,
    openai_api_key=api_key_openai
)

class VerifyAction(BaseModel):
    appropriate: bool = Field(description="Whether the action is appropriate or not")

class AvailabilityDecision(BaseModel):
    available_from: Optional[str] = Field(default=None, description="ISO timestamp when the agent becomes available, if None, available immediately")
    reasoning: str = Field(description="Brief explanation for the availability decision")

class Agent(BaseModel):
    """Base class for all agents"""
    id: str = Field(description="Unique identifier for the agent")
    role: str = Field(description="Role of the agent")
    schedule: Optional[str] = Field(default="Full-time", description="Schedule of the agent")
    data: Optional[str] = Field(default=None, description="Data related to the agent")
    actions: List[str] = Field(default_factory=list, description="List of actions the agent can perform")
    llm: Optional[Any] = Field(default=None, description="LLM instance to use for this agent (optional). If not provided, pass llm into methods.")
    prompt_manager: Optional[PromptManager] = Field(default=None, description="PromptManager instance for accessing prompts and fewshots")
    tool_manager: Optional[ToolManager] = Field(default=None, description="Tool manager for accessing tools")
    available_from: Optional[str] = Field(default=None, description="Timestamp when the agent becomes available, if None, available immediately")
    work_memory: List[Dict[str, Any]] = Field(default_factory=list, description="What the agent has done in the recent past")

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def generate_data_onfly(self, organization_context, data_model: BaseModel, llm, reflect=True):
        data_attributes = data_model.model_fields.keys()
        # create string from list of data attributes
        data_attributes = ', '.join(data_attributes)
        # fetch prompt template
        sys_prompt = self.prompt_manager.get_prompt(file_name="ext_agent_data_generation", prompt_name="data_generation", data="template")
        # allow passing llm per-call or use the agent's llm
        if llm is None:
            llm = self.llm
        if llm is None:
            raise ValueError("No llm provided: pass an llm to generate_data_onfly or set agent.llm")

        prompt = PromptTemplate.from_template(sys_prompt)

        # tool call 
        if self.tool_manager and self.tool_manager.use_tools:
            tools = self.tool_manager.get_stats_tool(name='get_resource_data_stats')
            llm_with_tools = llm.bind_tools(tools)
            # separate tool call prompt
            tool_prompt = self.prompt_manager.get_prompt(file_name="ext_agent_data_generation", prompt_name="data_generation_tool", data="template")
            tool_prompt = PromptTemplate.from_template(tool_prompt)
            tool_chain = tool_prompt | llm_with_tools
            tool_output = tool_chain.invoke({"organization_context": organization_context, "data_attributes": data_attributes})
            tool_calls = getattr(tool_output, 'tool_calls', None)
            tool_messages = []
            if tool_calls:
                for tool_call in tool_calls:
                    selected_tool = next((tool for tool in tools if tool.name.lower() == tool_call["name"].lower()), None)
                    if selected_tool is not None:
                        try:
                            tool_msg = selected_tool.invoke(tool_call)
                            tool_messages.append(tool_msg)
                        except Exception as e:
                            print(f'Error occurred while invoking tool {tool_call["name"]}: {e}')
                    else:
                        print(f'Tool: {tool_call["name"]} not found in available tools.')
                    
        else:
            llm_with_tools = llm
            tool_messages = []
            tool_output = None

        structured_llm = llm_with_tools.with_structured_output(data_model)

        # give tool messages to final data generation
        base_messages = prompt.invoke({"role": self.role, "data_attributes": data_attributes, "organization_context": organization_context})
        if tool_messages:
            full_messages = [HumanMessage(content=base_messages.text), tool_output] + tool_messages
            final_output = structured_llm.invoke(full_messages)
        else:
            final_output = structured_llm.invoke(base_messages)

        data =  ", ".join(f"{k}: {v}" for k, v in final_output.model_dump().items())

        if not reflect:
            self.data = data
            return
        
        # reflect if needed

        sys_prompt_reflect = self.prompt_manager.get_prompt(file_name="ext_agent_data_generation", prompt_name="data_generation_reflect", data="template")
        prompt_reflect = PromptTemplate.from_template(sys_prompt_reflect)
        chain_reflect = prompt_reflect | structured_llm
        data_reflect = chain_reflect.invoke({"role": self.role, "data_attributes": data_attributes, "data": data,  "organization_context": organization_context})
        data_reflect = ", ".join(f"{k}: {v}" for k, v in data_reflect.model_dump().items())
        self.data = data_reflect
        

    def generate_data(self, organization_context, org_agents, data_model: BaseModel, llm: Optional[Any] = None, reflect=False):
        data_attributes = data_model.model_fields.keys()
        # create string from list of data attributes
        data_attributes = ', '.join(data_attributes)
        # add random text to the prompt
        # create prompt for data
        sys_prompt = self.prompt_manager.get_prompt(file_name="org_agent_data_generation", prompt_name="data_generation", data="template")
        # allow passing llm per-call or use the agent's llm
        if llm is None:
            llm = self.llm
        if llm is None:
            raise ValueError("No llm provided: pass an llm to generate_data or set agent.llm")

        prompt = PromptTemplate.from_template(sys_prompt)
        # tool call
        if self.tool_manager and self.tool_manager.use_tools:
            tools = self.tool_manager.get_stats_tool(name='get_resource_data_stats')
            llm_with_tools = llm.bind_tools(tools)
            # separate tool call prompt
            tool_prompt = self.prompt_manager.get_prompt(file_name="org_agent_data_generation", prompt_name="data_generation_tool", data="template")
            tool_prompt = PromptTemplate.from_template(tool_prompt)
            tool_chain = tool_prompt | llm_with_tools
            tool_output = tool_chain.invoke({"data_attributes": data_attributes})
            tool_calls = getattr(tool_output, 'tool_calls', None)
            tool_messages = []
            if tool_calls:
                for tool_call in tool_calls:
                    selected_tool = next((tool for tool in tools if tool.name.lower() == tool_call["name"].lower()), None)
                    if selected_tool is not None:
                        try:
                            tool_msg = selected_tool.invoke(tool_call)
                            tool_messages.append(tool_msg)
                            #print(f'Tool: {tool_call["name"]} is being called by llm with arguments {tool_call.get("args", tool_call.get("arguments", {}))}')
                            #print(f'Tool message: {tool_msg.text}')
                        except Exception as e:
                            print(f'Error occurred while invoking tool {tool_call["name"]}: {e}')
                    else:
                        print(f'Tool: {tool_call["name"]} not found in available tools.')
                    
        else:
            llm_with_tools = llm
            tool_messages = []
            tool_output = None

        structured_llm = llm_with_tools.with_structured_output(data_model)

        # give tool messages to final data generation
        base_messages = prompt.invoke({"role": self.role, "data_attributes": data_attributes, "organization_context": organization_context})
        if tool_messages:
            full_messages = [HumanMessage(content=base_messages.text), tool_output] + tool_messages
            final_output = structured_llm.invoke(full_messages)
        else:
            final_output = structured_llm.invoke(base_messages)

        data =  ", ".join(f"{k}: {v}" for k, v in final_output.model_dump().items())

        if not reflect:
            self.data = data
            return

        # reflect if needed
        sys_prompt_reflect = self.prompt_manager.get_prompt(file_name="org_agent_data_generation", prompt_name="data_generation_reflect", data="template")
        prompt_reflect = PromptTemplate.from_template(sys_prompt_reflect)
        chain_reflect = prompt_reflect | structured_llm
        data_reflect = chain_reflect.invoke({"role": self.role, "data_attributes": data_attributes, "data": data,  "organization_context": organization_context, "org_agents": org_agents})
    
        data_reflect = ", ".join(f"{k}: {v}" for k, v in data_reflect.model_dump().items())
        self.data = data_reflect

    
    def _create_action_prompt(self):
        """Create prompt for the agent using PromptManager if available"""
        # Try to use PromptManager first
        if not self.prompt_manager:
            raise ValueError("PromptManager is required for agent operation")
            
        # Get the template from the YAML file
        # Default to agent_action, but can be customized later
        template = self.prompt_manager.get_prompt('agent_action', 'agent_action', 'template')
        
        # Replace agent-specific placeholders with actual values
        role = self.role
        actions = ', '.join(self.actions)
        data = self.data if self.data else "No data available"
        
        # Replace placeholders in the template
        template = template.replace('{role}', role).replace('{data}', data).replace('{actions}', actions)
            
        return template
    
    def _create_tool_call_prompt(self):
        """Create prompt for the agent to call tools"""
        role = self.role
        actions = ', '.join(self.actions)
        data = self.data if self.data else "No data available"
        prompt = self.prompt_manager.get_prompt(file_name="agent_action", prompt_name="agent_action_tool", data="template")
        prompt = prompt.replace('{role}', role).replace('{data}', data).replace('{actions}', actions)
        
        return prompt
    
    def _take_action(self, output_model, llm: Optional[Any], process_summary: str, process_state: str, action_data_mapping: Dict[str, List[str]], current_time: str, event_data_attributes: List[str], case_data_attributes: List[str], message: str, conventions: List[str]):
        """
        Act based on the process summary, process state, action data mapping, current time, event data attributes, and case data attributes.

        Args:
            process_summary (str): Summary of the process.
            process_state (str): Current state of the process.
            action_data_mapping (Dict[str, List[str]]): Mapping of actions to their required data fields.
            current_time (str): Current timestamp.
            event_data_attributes (List[str]): List of event data attributes.
            case_data_attributes (List[str]): List of case data attributes.

        Returns:
            AgentOutput: Structured output containing the action taken by the agent.
        """
        # resolve llm: use passed llm or agent.llm
        if llm is None:
            llm = self.llm
        if llm is None:
            raise ValueError("No llm provided: pass an llm to act/_take_action or set agent.llm")

        prompt = self._create_action_prompt()
        
        # Load fewshots on-demand from PromptManager
        fewshot_examples = []
        if self.prompt_manager.use_fewshots:
            fewshots = self.prompt_manager.get_fewshots('agent_action')
            # Check if fewshots were loaded successfully (not an error list)
            if fewshots and not (len(fewshots) == 1 and 'error' in fewshots[0]):
                fewshot_examples = fewshots

        if fewshot_examples:
            # example prompt template
            example_prompt = PromptTemplate(
                input_variables=["role", "data", "actions", "process_summary", "process_state", "action_data_mapping", "event_data_attribute", "case_data_attribute", "conventions", "message", "current_time", "output"],
                template="Role: {role}\nData: {data}\nActions: {actions}\nProcess Summary: {process_summary}\nProcess State: {process_state}\nAction Data Mapping: {action_data_mapping}\nEvent Data Attributes: {event_data_attribute}\nCase Data Attributes: {case_data_attribute}\nConventions: {conventions}\nMessage: {message}\nCurrent Time: {current_time}\nOutput: {output}\n---\n"
            )
            final_prompt = FewShotPromptTemplate(
                examples=fewshot_examples,
                example_prompt=example_prompt,
                input_variables=["process_summary", "process_state", "action_data_mapping", "event_data_attribute", "case_data_attribute", "conventions", "message", "current_time"],
                prefix=prompt
            )

        else:
            final_prompt = PromptTemplate.from_template(prompt)

        tool_messages = []
        # Tool call
        if self.tool_manager and self.tool_manager.use_tools:
            tools = self.tool_manager.get_stats_tool(name=['get_event_data_stats', 'get_case_data_stats', 'get_event_data_stats_by_resource_group']) + self.tool_manager.get_generate_tool()
            llm_with_tools = llm.bind_tools(tools)
            # separate tool call prompt
            tool_prompt = self._create_tool_call_prompt()
            tool_prompt = PromptTemplate.from_template(tool_prompt)
            tool_chain = tool_prompt | llm_with_tools
            tool_output = tool_chain.invoke({
                "process_summary": process_summary,
                "process_state": process_state,
                "action_data_mapping": '; '.join([f"{action} → {', '.join(fields)}" for action, fields in action_data_mapping.items()]),
                "current_time": current_time,
                "event_data_attributes": "; ".join(f"{k} → {v}" for k, v in event_data_attributes.items()),
                "case_data_attributes": '; '.join(f"{k} → {v}" for k, v in case_data_attributes.items()),
                "message": message,
                "conventions":  '; '.join(conventions)
            })
        
            # get tool calls output messages
            tool_calls = getattr(tool_output, 'tool_calls', None)
            if tool_calls:
                for tool_call in tool_calls:
                    selected_tool = next((tool for tool in tools if tool.name.lower() == tool_call["name"].lower()), None)
                    if selected_tool is not None:
                        #print(f'Tool: {tool_call["name"]} is being called by llm with arguments {tool_call.get("args", tool_call.get("arguments", {}))}')
                        try:
                            tool_msg = selected_tool.invoke(tool_call)
                            #print(f'Tool message: {tool_msg.content}')
                            tool_messages.append(tool_msg)
                        except Exception as e:
                            print(f'Error occurred while invoking tool {tool_call["name"]}: {e}')
                    else:
                        print(f'Tool: {tool_call["name"]} not found in available tools.')
                    
        else:
            llm_with_tools = llm
            tool_output = None

        # Add tool call messages to generate final output.
        structured_llm = llm_with_tools.with_structured_output(output_model)
        base_messages = final_prompt.invoke({
            "process_summary": process_summary,
            "process_state": process_state,
            "action_data_mapping": '; '.join([f"{action} → {', '.join(fields)}" for action, fields in action_data_mapping.items()]),
            "current_time": current_time,
            "event_data_attributes": "; ".join(f"{k} → {v}" for k, v in event_data_attributes.items()),
            "case_data_attributes": '; '.join(f"{k} → {v}" for k, v in case_data_attributes.items()),
            "message": message,
            "conventions":  '; '.join(conventions)
        })
        full_messages = [HumanMessage(content=base_messages.text), tool_output] + tool_messages
        agent_outputs = structured_llm.invoke(full_messages)

        return agent_outputs, tool_messages
    
    def _create_modify_prompt(self):
        """Create modification prompt for the agent"""
        role = self.role
        actions = ', '.join(self.actions)
        data = self.data if self.data else "No data available"
        modify_prompt = self.prompt_manager.get_prompt(file_name="agent_action", prompt_name="agent_action_modify", data="template")
        modify_prompt = modify_prompt.replace('{role}', role).replace('{data}', data).replace('{actions}', actions)
        return modify_prompt
    
    def _decide_availability(self, llm: Optional[Any], memory_summary: str, recent_actions: str, current_time: str, memory_retention_days: int):
        """Decide agent availability based on schedule, memory, recent actions, and current time."""

        if llm is None:
            llm = self.llm
        if llm is None:
            raise ValueError("No llm provided: pass an llm to _decide_availability or set agent.llm")

        prompt_template = self.prompt_manager.get_prompt(file_name="agent_availability", prompt_name="agent_availability", data="template")
        prompt = PromptTemplate.from_template(prompt_template)
        structured_llm = llm.with_structured_output(AvailabilityDecision)

        messages = prompt.invoke({
            "role": self.role,
            "data": self.data if self.data else "No data available",
            "schedule": self.schedule,
            "memory_summary": memory_summary,
            "recent_actions": recent_actions,
            "current_time": current_time,
            "memory_retention_days": memory_retention_days
        })

        availability_decision = structured_llm.invoke(messages)

        self.available_from = availability_decision.available_from

    @staticmethod
    def _safe_parse_iso_datetime(timestamp: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp defensively; return None for invalid values."""
        if not timestamp:
            return None
        try:
            return datetime.fromisoformat(timestamp)
        except (TypeError, ValueError):
            return None
    
    def _update_work_memory(self, actions, memory_retention_days, current_time):
        """Update the agent's work memory with recent actions.
            Note: this work memory is only in-memory, not persisted, and contain only work in the recent past (retention days)"""
        # get current work memory
        work_memory = self.work_memory # List of dicts {process_id, action, start_timestamp, end_timestamp}
        # add new actions to work memory
        for action in actions.agent_outputs:
            work_memory.append({
                "process_id": action.process_id,
                "action": action.action,
                "start_timestamp": action.start_timestamp,
                "end_timestamp": action.end_timestamp,
            })
        # remove old actions beyond retention days
        current_dt = self._safe_parse_iso_datetime(current_time)
        if current_dt is None:
            self.work_memory = work_memory
            return

        retention_cutoff = current_dt - timedelta(days=memory_retention_days)
        self.work_memory = [
            entry
            for entry in work_memory
            if (start_dt := self._safe_parse_iso_datetime(entry.get("start_timestamp"))) is not None
            and start_dt >= retention_cutoff
        ]

    def _get_work_memory_summary(self):
        """Get a summary of the agent's work memory over the retention period.
         Returns a string summary of recent actions: {num_handled_processes: X, num_handled_events: Y, total_time_spent: Z hours, average_time_per_day: W hours}"""
        total_time = timedelta(0)
        for entry in self.work_memory:
            start_dt = self._safe_parse_iso_datetime(entry.get("start_timestamp"))
            end_dt = self._safe_parse_iso_datetime(entry.get("end_timestamp"))
            if start_dt is None or end_dt is None or end_dt < start_dt:
                continue
            total_time += (end_dt - start_dt)

        stats = {
            "num_handled_processes": len(set(entry["process_id"] for entry in self.work_memory)),
            "num_handled_events": len(self.work_memory),
            "total_time_spent_hours_on_the_tasks_alone": total_time.total_seconds() / 3600 if total_time else 0.0,
        }
        return ', '.join(f"{key}: {value}" for key, value in stats.items())

    def act(self, output_model, llm: Optional[Any] = None, process_summary: str = "", process_state: str = "", action_data_mapping: Dict[str, List[str]] = None, current_time: str = "", event_data_attributes: List[str] = None, case_data_attributes: List[str] = None, message: str = "", conventions: str = "", reflect=True, enable_availability=True, memory_retention_days: int = 7):
        
        # STEP 1: Agent  takes action    
        actions, tool_messages = self._take_action(
            llm=llm,
            process_summary=process_summary,
            process_state=process_state,
            action_data_mapping=action_data_mapping,
            current_time=current_time,
            event_data_attributes=event_data_attributes,
            case_data_attributes=case_data_attributes,
            message=message,
            output_model=output_model,
            conventions=conventions
        )
        # STEP 2: Reflect on action if needed
        if reflect:
            modify_prompt = self._create_modify_prompt()
            # resolve llm for verification
            if llm is None:
                llm = self.llm
            if llm is None:
                raise ValueError("No llm provided for verification: pass llm to act or set agent.llm")

            structured_llm = llm.with_structured_output(output_model)
            prompt = PromptTemplate.from_template(modify_prompt)
            chain = prompt | structured_llm

            tool_messages_str = [tool_msg.content for tool_msg in tool_messages]

            modified_agent_outputs = []
            for action in actions.agent_outputs:
                start_timestamp_str = extract_time_parts(action.start_timestamp)
                end_timestamp_str = extract_time_parts(action.end_timestamp)
                action_string = (
                    f'Process_id: {action.process_id}, '
                    f'action: {action.action}, '
                    f'timestamp: You started this action {start_timestamp_str} and finished this action {end_timestamp_str}, '
                    f'case_data: {", ".join([f"{key}: {value}" for key, value in (action.case_data.model_dump().items() if action.case_data else []) if value is not None])}, '
                    f'event_data: {", ".join([f"{key}: {value}" for key, value in (action.event_data.model_dump().items() if action.event_data else []) if value is not None])}, '
                    f'message: {action.message}, '
                    f'end_action: {action.end_action}'
                )
                modified_agent_output = chain.invoke({
                    "process_summary": process_summary,
                    "process_state": process_state,
                    "current_time": current_time,
                    "previous_action": action_string,
                    "conventions": conventions,
                    "event_data_attributes": "; ".join(f"{k} → {v}" for k, v in event_data_attributes.items()),
                    "case_data_attributes": '; '.join(f"{k} → {v}" for k, v in case_data_attributes.items()),
                    "tool_messages": "\n".join(tool_messages_str)
                })
                
                # Check if agent_outputs contains any elements before accessing [0]
                if modified_agent_output.agent_outputs and len(modified_agent_output.agent_outputs) > 0:
                    modified_agent_outputs.append(modified_agent_output.agent_outputs[0])
                else:
                    # If no modified output is generated, keep the original action
                    modified_agent_outputs.append(action)

            actions.agent_outputs = modified_agent_outputs

        # STEP 3: Agent decides availability
        if enable_availability:
            self._update_work_memory(actions, memory_retention_days, current_time)
            memory_summary = self._get_work_memory_summary()
            recent_actions = "; ".join(
                f'Process_id: {entry["process_id"]}, action: {entry["action"]}, start_timestamp: {entry["start_timestamp"]}, end_timestamp: {entry["end_timestamp"]}'
                for entry in self.work_memory
            )
            self._decide_availability(
                llm=llm,
                memory_summary=memory_summary,
                recent_actions=recent_actions,
                current_time=current_time,
                memory_retention_days=memory_retention_days,
            )
            print(f'Memory summary for agent {self.id}: {memory_summary}')
            print(f'Current time is {current_time} and agent {self.id} will be available from {self.available_from}')
        
        return actions, tool_messages


    

