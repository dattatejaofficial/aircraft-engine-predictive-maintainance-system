import sys
from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.database.database_manager import DatabaseManager
from predictivesystem.database.database_schema import TRAIN_FEATURES, TEST_FEATURES, TEST_TARGETS

class DatabaseInitializer:
    def __init__(self, db_manager : DatabaseManager):
        self.db_manager = db_manager
    
    def initialize_database(self):
        try:
            logging.info("Initializing Database")
            self.db_manager.connect()

            self.db_manager.execute(query=TRAIN_FEATURES)
            logging.info("Created Table for Train Features")

            self.db_manager.execute(query=TEST_FEATURES)
            logging.info("Created Table for Test Features")

            self.db_manager.execute(query=TEST_TARGETS)
            logging.info("Created Table for Test Target")

            self.db_manager.commit()
            self.db_manager.close()
            logging.info("Disconnecting Database")
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)