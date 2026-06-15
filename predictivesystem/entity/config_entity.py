import yaml

class ConfigurationManager:
    def __init__(self, config_filepath = "configs/config.yaml"):
        with open(config_filepath, 'r') as file:
            self.config = yaml.safe_load(file)
    
    @property
    def logging_config(self):
        return self.config['logging']