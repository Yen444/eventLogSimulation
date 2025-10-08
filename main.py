from dotenv import load_dotenv
import os
import random
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')


def main():
  
    from simulation_src import SimulationEngine

    # write arrivals to csv file
    # Version 0: female resource takes more time
    # Version 01: male resource takes more time than female, invoice should be cleared at least one week after the corresponding goods receipt has been recorded.

    start_time = '2018-01-01T00:00:00+00:00'
    num_traces = 500

    # P2P cleaned process

    # Average requests per month (index 0 = Jan)
    monthly_rates = [3.500000, 7.500000, 5.500000, 4.333333, 3.916667, 5.750000, 
                    3.916667, 4.083333, 6.083333, 5.416667, 5.666667, 2.000000]

    # Average requests per weekday (0=Mon, ..., 6=Sun)
    weekday_rates = [2.176471, 2.039216, 3.352941, 3.333333, 2.411765, 0.196078, 0.058824]

    # Average requests per hour (0–23)
    hourly_rates = [0, 0, 0.052326, 0.139535, 0.232558, 0.325581, 0.430233, 0.453488,
                    0.401163, 0.366279, 0.441860, 0.319767, 0.406977, 0.215116, 0.098837, 0.040698,
                    0.011628, 0.005814, 0.005814, 0.075581, 0, 0, 0, 0]

    # Base rate (can be scaled up later)
    BASE_RATE_HOUR = 0.06 # there is one request every 100 hours on average

    from simulation_src.utils import simulate_arrivals

    # ensure output directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('protocols', exist_ok=True)
    os.makedirs('db', exist_ok=True)

    # simulate arrivals
    arrivals, inter_arrival_time = simulate_arrivals(start_time=start_time,
                                 monthly_rate=monthly_rates,
                                 weekday_rate=weekday_rates,
                                 hourly_rate=hourly_rates,
                                 base_rate=BASE_RATE_HOUR,
                                 max_events=num_traces)

    arrivals_csv_path = os.path.join('logs', 'arrivals_p2p_3WayMatchingEC.csv')
    with open(arrivals_csv_path, 'w', encoding='utf-8') as f:
        f.write('timestamp,lamda\n')
        for arrival in arrivals:
            f.write(f'{arrival[0]},{arrival[1]}\n')

    # paths
    simulation_config_path = './configs/config_p2p_3WayMatchingEC.json'
    log_path = './logs/p2p_3WayMatchingEC_fewshots_01.csv'
    protocol_path = './protocols/p2p_3WayMatchingEC_fewshots_01.csv'
    db_path = './db'

    # instantiate the LLM lazily to avoid import-time failures
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', temperature=0.5)
    except Exception:
        print('Warning: Failed to initialize ChatGoogleGenerativeAI. Make sure required LLM packages and API keys are installed and configured.')
        raise
    
    # create and run the simulation engine
    simulation_engine = SimulationEngine(llm=llm_mid, config_path=simulation_config_path, log_path=log_path, protocol_path=protocol_path, db_path=db_path, arrivals=arrivals)
    simulation_engine.run(start_time=start_time, num_traces=num_traces, inter_arrival_time=inter_arrival_time)

if __name__ == "__main__":
    main()
