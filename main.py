from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm_low = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0)
llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', temperature=0.5)
llm_high = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)

def main():
    from simulation_src import SimulationEngine

    
    simulation_config_path = './configs/config_default.json'
    log_path = './logs/p2p_3WayMatchingEC.csv'
    simulation_engine = SimulationEngine(llm=llm_mid, config_path=simulation_config_path, log_path=log_path)
    start_time = '2018-01-01T00:00:00+00:00'
    end_time = '2019-01-01T00:00:00+00:00'
    inter_arrival_time = 3600 * 18
    simulation_engine.run(start_time, end_time, inter_arrival_time)


if __name__ == "__main__":
    main()
