import sys
from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.database.database_manager import DatabaseManager
from predictivesystem.database.database_schema import TEST_RUL_SCHEMA

class DatabaseInitializer:
    def __init__(self, db_manager : DatabaseManager):
        self.db_manager = db_manager
    
    def initialize_database(self):
        pass