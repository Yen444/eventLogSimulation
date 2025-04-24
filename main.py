def main():
    from simulation import SimulationEngine

    config_file = './configs/configs.json'
    bpmn_file = './bpmn_files/Data_set/3-8.bpmn'
    log_file = './logs/insurance_claim_process.csv'

    simulation_engine = SimulationEngine(config_path=config_file, bpmn_path=bpmn_file, log_path=log_file)
    start_time = '2023-10-01T00:00:00+00:00'
    end_time = '2023-10-02T00:00:00+00:00'
    inter_arrival_time = 360  # in seconds
    simulation_engine.run(start_time, end_time, inter_arrival_time)

if __name__ == "__main__":
    main()
