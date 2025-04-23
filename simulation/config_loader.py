class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        import json
        with open(self.config_file, 'r') as file:
            return json.load(file)

    def get_config(self):
        return self.config
    def get_external_factors(self):
        return self.config.get('external_factors', [])
    def get_demographics(self):
        return self.config.get('demographics', [])
    def get_num_agents(self):
        return self.config.get('num_agents', {})
    def get_agent_actions(self):
        return self.config.get('process_data').get('agent_actions', {})
    def get_process_summary(self):
        return self.config.get('process_data').get('process_summary', '')
    def get_start_end_events(self):
        return self.config.get('process_data').get('start_end_events', {})
    def get_activities_flow(self):
        return self.config.get('process_data').get('activities_flow', {})