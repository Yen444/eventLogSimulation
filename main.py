from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm_low = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0)
llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5)
llm_high = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)

def main():
    from simulation import SimulationEngine

    bpmn_file = './bpmn_files/Data_set/3-8.bpmn'
    extracted_config_file = './configs/extracted_configs.json'
    user_config_file = './configs/user_configs.json'
    simulation_config_file = './configs/configs.json'
    log_file = './logs/insurance_claim_process_05.csv'

    simulation_engine = SimulationEngine(
        bpmn_path=bpmn_file,
        extracted_config_path=extracted_config_file,
        user_config_path=user_config_file,
        simulation_config_path=simulation_config_file,
        log_path=log_file
    )
    start_time = '2023-10-01T00:00:00+00:00'
    end_time = '2023-10-02T00:00:00+00:00'
    inter_arrival_time = 360  # in seconds
    simulation_engine.run(start_time, end_time, inter_arrival_time)


if __name__ == "__main__":
    main()
