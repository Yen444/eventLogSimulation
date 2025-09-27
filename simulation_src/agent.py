from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any, Type
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


from .llm_utils import generate_random_text_with_mem_chatbot
from .utils import extract_time_parts
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

class VerifyAction(BaseModel):
    appropriate: bool = Field(description="Whether the action is appropriate or not")

class Agent(BaseModel):
    """Base class for all agents"""
    id: str = Field(description="Unique identifier for the agent")
    role: str = Field(description="Role of the agent")
    data: Optional[str] = Field(default=None, description="Data related to the agent")
    actions: List[str] = Field(default_factory=list, description="List of actions the agent can perform")
    action_fewshots: List[Dict[str, Any]] = Field(default_factory=list, description="Fewshot examples for the agents taking suitable actions")

    def generate_data_onfly(self, organization_context, data_model: BaseModel, llm, reflect=True):
        data_attributes = data_model.model_fields.keys()
        # create string from list of data attributes
        data_attributes = ', '.join(data_attributes)
        # add random text to the prompt
        random_text = generate_random_text_with_mem_chatbot()
        # create prompt for data
        sys_prompt = """
        You are tasked with generating realistic and coherent data for a person with the role of {role}.
        This individual operates outside the organization but interacts with it within a business process context.

        Organizational context:
        {organization_context}

        Please generate plausible values for the following attributes: {data_attributes}.

        Guidelines:
        - Ensure all values are internally consistent and appropriate for the specified role and organizational context.
        - Emphasize diversity, plausibility, and details specific to the role.
        
        Return the output in a clearly structured format, listing each attribute and its corresponding value:
        - <attribute>: <value>
        - <attribute>: <value>
        - ...
        """

        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = llm.with_structured_output(data_model)
        chain = prompt | structured_llm
        data = chain.invoke({"role": self.role, "data_attributes": data_attributes, "organization_context": organization_context})
        data =  ", ".join(f"{k}: {v}" for k, v in data.model_dump().items())
        if not reflect:
            # write to data values
            self.data = data
            return

        sys_prompt_reflect = """
        You are tasked with reviewing and refining personal data for a person with the role of {role} who interacts with the organization externally within a business process context.

        Organizational context:
        {organization_context}

        You have been provided with initial data that may contain generic, unrealistic, or placeholder values. Your goal is to revise these entries to ensure the data is realistic, coherent, and appropriate for the given role and organizational context.

        Guidelines:
        - Avoid generic placeholders such as "John Doe", "Jane Doe", "A. N. Other", "Test User", or similarly artificial values.
        - Ensure all attributes are plausible, unique, and reflect real-world diversity, relevant to the person's role.
        - Maintain consistency with the organizational culture and business process context.
        - Ensure that the generated data does not represent any biases or stereotypes related to gender, ethnicity, age, or other personal characteristics. Strive for fair and balanced representation in all attributes.

        The data should cover the following attributes: {data_attributes}.

        Initial data to review and refine:
        {data}

        Keep attribute-value pairs that are already realistic and specific; only update those that are too generic, inconsistent, or implausible.

        Return the final output clearly formatted, listing each attribute with its corresponding value:
        - <attribute>: <value>
        - <attribute>: <value>
        - ...
        """


        prompt_reflect = PromptTemplate.from_template(sys_prompt_reflect)
        chain_reflect = prompt_reflect | structured_llm
        data_reflect = chain_reflect.invoke({"role": self.role, "data_attributes": data_attributes, "data": data,  "organization_context": organization_context})
    
        data_reflect = ", ".join(f"{k}: {v}" for k, v in data_reflect.model_dump().items())
        self.data = data_reflect
        
        #print(f'First generated data: {data}, role: {self.role}')
        #print(f'Regenerated data: {data_reflect}, role: {self.role}')

    def generate_data(self, organization_context, org_agents, data_model: BaseModel, llm, reflect=True):
        data_attributes = data_model.model_fields.keys()
        # create string from list of data attributes
        data_attributes = ', '.join(data_attributes)
        # add random text to the prompt
        random_text = generate_random_text_with_mem_chatbot()
        # create prompt for data
        sys_prompt = """
        You are tasked with generating realistic and coherent data for a person with the role of {role} within an organization.

        The organizational context is as follows: {organization_context}.

        Generate plausible values for the following attributes: {data_attributes}. Ensure the values are internally consistent and appropriate for the given role and organizational context.

        When generating this person's information, take into account the existing members of the organization to avoid unrealistic overlaps. For instance, if there are already two employees named Alice who are 30 years old, avoid adding a third with the same name and age.

        Existing organization members:
        {org_agents}

        {random_text}

        Return the output in a clearly structured format, listing each attribute and its corresponding value:
        - <attribute>: <value>
        - <attribute>: <value>
        - ...
        """

        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = llm.with_structured_output(data_model)
        chain = prompt | structured_llm
        data = chain.invoke({"role": self.role, "data_attributes": data_attributes, "organization_context": organization_context, "org_agents": org_agents, "random_text": random_text})
        data =  ", ".join(f"{k}: {v}" for k, v in data.model_dump().items())
        if not reflect:
            # write to data values
            self.data = data
            return

        sys_prompt_reflect = """
        You are tasked with reviewing and refining the personal data of a person with the role of {role} within an organization.

        Organizational context:
        {organization_context}

        Existing organization members:
        {org_agents}

        You are provided with initial data that may contain generic, unrealistic, or placeholder-like values. Your goal is to revise any such entries to produce realistic, coherent, and contextually appropriate information that fits within the organization.

        Guidelines:
        - **AVOID GENERIC PLACEHOLDERS**: Do not use names like "John Doe", "Jane Doe", "A. N. Other", "Test User", or any similarly artificial values.
        - **ENSURE REALISM & UNIQUENESS**: All attributes should reflect plausible real-world data. Names, ages, job details, etc., should be context-aware and diverse.
        - **RESPECT CONTEXT**: Ensure that the attributes make sense given the person's role, the organization's culture, ect.
        - **AVOID REPETITION**: Avoid duplications with existing employees (e.g., is it plausible for two employees both named Alice Bob and aged 30?)
        - - **AVOID BIAS/STEREOTYPE**: Ensure that the generated data does not represent any biases or stereotypes related to gender, ethnicity, age, or other personal characteristics. Strive for fair and balanced representation in all attributes.

        The data should include the following attributes: {data_attributes}.

        Here is the initial data to be reviewed and refined:
        {data}

        Keep attribute-value pairs that are already realistic and specific. Update or replace only the ones that are too generic, inconsistent, or implausible.

        Return the final output in a structured format, listing each attribute and its corresponding value:
        - <attribute>: <value>
        - <attribute>: <value>
        - ...
        """


        prompt_reflect = PromptTemplate.from_template(sys_prompt_reflect)
        chain_reflect = prompt_reflect | structured_llm
        data_reflect = chain_reflect.invoke({"role": self.role, "data_attributes": data_attributes, "data": data,  "organization_context": organization_context, "org_agents": org_agents})
    
        data_reflect = ", ".join(f"{k}: {v}" for k, v in data_reflect.model_dump().items())
        self.data = data_reflect
        
        #print(f'First generated data: {data}, role: {self.role}')
        #print(f'Regenerated data: {data_reflect}, role: {self.role}')

    
    def _create_action_prompt(self):
        """Create prompt for the agent"""
        role = self.role
        actions = ', '.join(self.actions)
        data = self.data if self.data else "No data available"
        prompt = f"""
        You are an agent participating in a business process.

        Process Summary: {{process_summary}}  
        Your Role: {role}  
        Your Profile: {data}  
        Available Actions: {actions}  

        Each action may require specific data. Below is a mapping of actions to their required data fields:  
        Action Data Mapping: {{action_data_mapping}} 

        You are assigned to the following current process instance with the message from the coordinator:
        Current Process State: {{process_state}}
        Message: {{message}}
        Timestamp: {{current_time}} 
           

        Based on the above information, decide whether you will take any action for the assigned process.  
        If you will take any action, specify the process_id, the action, the start and end timestamp at which you started and finished your action, and include necessary event and case data attributes.
        
        Some actions are easy to do, e.g. submit a claim, but some actions take time, e.g. now you need to deliver the goods but the order is only made 5 minutues before, you need to wait for the order to be done first.
        Therefore, use your logical thinking basing on the process state to take suitable action.

        When you take an action, consider the following conventions, your action must reflect the following conventions: {{conventions}}

        Event Data Attributes: {{event_data_attributes}}
        Case Data Attributes: {{case_data_attributes}}

        You must output all values for the event data attributes.
        For case data attributes, only output the values that your action requires, if any. 

        **Please use the following structured format for each action taken:**

        - process_id: ```<process_id>```  
        - action: ```<your_chosen_action>```  
        - start_timestamp: ```<start_timestamp_of_action>```
        - end_timestamp: ```<end_timestamp_of_action>```  
        - case_data: ```<case_data_attribute_1>: <case_data_value_1>, <case_data_attribute_2>: <case_data_value_2>, ...```  
        - event_data: ```<event_data_attribute_1>: <event_data_value_1>, <event_data_attribute_2>: <event_data_value_2>, ...```
        - message: ```<your_message>```
        - end_action: ```<True> or <False>```

        **If you think you can't finish any action**
        - Leave the `action` field and other fields empty.
        
        **Message should be a short description of what you did, your decision, or any other relevant information. If you can't take the action at the moment (because you have to wait for something or you're overloaded, etc.), specify the reasons.**

        **Only output the data needed for the action you will take, which is specified in the action data mapping above, leave other attributes empty if it is not required for that action**

        **Output a suitable timestamp for the action you will finish. The timestamp must satisfy all of the following conditions:**
        - It must not be earlier than the timestamp you're assigned to.

        **If the current process state is empty, which means you're the first one to take action to initialize the process"
        - Leave the process_id empty and output the other fields like described.
        - The end_timestamp of your initializing action MUST be identical to the timestamp at which you are assigned this action.

        **If the current process state is not empty, which means there is already a process_id**
        - Output the correct process_id like the one given to you and output the other fields like described.

        **If you finish multiple actions:**
        - Repeat the above structured format for each action.

        **If you think the action you take can end the process:**
        - Output True for end_action.

        **You can take consecutive actions for one process and same actions for different processes, depending on the process and process states you're given.**

        **Read the process summary carefully, read the message from the coordinator, see what actions you can take, and think logically about what you can do next.**
        **Sometimes you're overloaded with so many other tasks or with your life. If you can't do your assigned task, just specifiy the reason in the message field and leave other fields empty.**
        **Be active: Even if you've only been assigned one action by the coordinator, take the initiative. Based on the process summary, your allowed actions, and the nature of those actions (e.g., quick tasks often performed in sequence), after taking your assigned action, identify and perform any consecutive actions that logically follow. Be proactive—move the process forward when it makes sense.**
        **Be aware: If you've been assigned an activity which you think is not suitable for that process (basing on the process summary and the process state), you can decide to take other action rather than the assigned one (if that action is suitable), or you can choose not to take any action at all.**
        """


        return prompt
    
    def _take_action(self, output_model, llm, process_summary: str, process_state: str, action_data_mapping: Dict[str, List[str]], current_time: str, event_data_attributes: List[str], case_data_attributes: List[str], update_interval:str, message: str, conventions: List[str]):
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
        prompt = self._create_action_prompt()
        fewshot_examples = self.action_fewshots

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


        structured_llm = llm.with_structured_output(output_model)
        chain = final_prompt | structured_llm
        agent_outputs = chain.invoke({
            "process_summary": process_summary,
            "process_state": process_state,
            "action_data_mapping": '; '.join([f"{action} → {', '.join(fields)}" for action, fields in action_data_mapping.items()]),
            "current_time": current_time,
            "event_data_attributes": "; ".join(f"{k} → {v}" for k, v in event_data_attributes.items()),
            "case_data_attributes": '; '.join(f"{k} → {v}" for k, v in case_data_attributes.items()),
            "message": message,
            "conventions":  '; '.join(conventions)
        })
        
        return agent_outputs
    
    def _create_verify_prompt(self):
        """Create verification prompt for the agent"""
        role = self.role
        actions = ', '.join(self.actions)
        data = self.data if self.data else "No data available"
        verify_prompt = f"""
        You are a reflective agent reviewing your previously decided actions in a business process.

        Here is a summary of the situation:
        - Process Summary: {{process_summary}}
        - Your Role: {role}
        - Your Profile: {data}
        - Available Actions: {actions}

        Contextual Information:
        - Current Timestamp: {{current_time}}
        - Current Process State: {{process_state}}

        You did plan to finish the following action:
        {{previous_action}}

        **Your Task:**
        Evaluate the *appropriateness* of the previously proposed action based *primarily* on the timestamps associated with that action. For the action, assess the following:

        1. **Are the timestamps appropriate:**  
        - Does the start and end timestamps fall within a typical and reasonable time window for work-related activities?  
        - Avoid suggesting actions during biologically or culturally inappropriate times, such as deep night hours, unless explicitly justified by the context.
        - Ask yourself questions like are you working on Saturday and Sunday?

        2. **Personal Feasibility:**  
        - Are you, given your role and demographic profile, realistically able to complete the action at the specified timestamps?  
        - Consider your energy levels, time availability, and workload distribution. If the timestamps are likely to hinder your ability to complete the action effectively, mark it as inappropriate.

        **For each action, determine whether its timestamps are both appropriate and personally feasible:**

        - If the action meets both criteria, output True.

        - If it fails either criterion, output False.

        **Output format:**
        - appropriate: ```<True or False>```

        """
        return verify_prompt

    def act(self, output_model, llm, process_summary: str, process_state: str, action_data_mapping: Dict[str, List[str]], current_time: str, event_data_attributes: List[str], case_data_attributes: List[str], message:str, conventions:str, update_interval: str = "1", reflect=True):
            
        actions = self._take_action(
            llm=llm,
            process_summary=process_summary,
            process_state=process_state,
            action_data_mapping=action_data_mapping,
            current_time=current_time,
            event_data_attributes=event_data_attributes,
            case_data_attributes=case_data_attributes,
            update_interval=update_interval,
            message=message,
            output_model=output_model,
            conventions=conventions
        )
        if not reflect:
            return actions
        verify_prompt = self._create_verify_prompt()
        structured_llm = llm.with_structured_output(VerifyAction)
        prompt = PromptTemplate.from_template(verify_prompt)
        chain = prompt | structured_llm
        to_remove = []

        for action in actions.agent_outputs:
            start_timestamp_str = extract_time_parts(action.start_timestamp)
            end_timestamp_str = extract_time_parts(action.end_timestamp)
            action_string = (
                f'Process_id: {action.process_id}, '
                f'action: {action.action}, '
                f'timestamp: You started this action {start_timestamp_str} and finished this action {end_timestamp_str}, '
                f'case_data: {", ".join([f"{key}: {value}" for key, value in (action.case_data.model_dump().items() if action.case_data else []) if value is not None])}, '
                f'event_data: {", ".join([f"{key}: {value}" for key, value in (action.event_data.model_dump().items() if action.event_data else []) if value is not None])}\n'
            )
            verified_agent_output = chain.invoke({
                "process_summary": process_summary,
                "process_state": process_state,
                "current_time": current_time,
                "previous_action": action_string
            })
            if verified_agent_output.appropriate == False:
                # remove action from the first actions
                print(f'ACTION IS REMOVED')
                to_remove.append(action)

        actions.agent_outputs = [action for action in actions.agent_outputs if action not in to_remove]
        return actions


    

