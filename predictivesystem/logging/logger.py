import os
import logging
from datetime import datetime
from predictivesystem.entity.config_entity import ConfigurationManager

config = ConfigurationManager()
logging_config = config.logging_config

LOG_FILE = f"{datetime.now().strftime(logging_config['timestamp_format'])}.log"
LOG_DIR = logging_config['logs_dir']

logs_path = os.path.join(os.getcwd(), LOG_DIR, LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format=logging_config['log_format'],
    level=getattr(logging, logging_config['log_level'].upper())
)