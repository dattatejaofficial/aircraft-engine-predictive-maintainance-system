import sys

from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import DatabaseConfig
from predictivesystem.database.database_manager import DatabaseManager
from predictivesystem.database.database_initializer import DatabaseInitializer

def main():
    try:
        db_config = DatabaseConfig()
        db_manager = DatabaseManager(config=db_config)
        db_initializer = DatabaseInitializer(db_manager=db_manager)

        db_initializer.initialize_database()

    except Exception as e:
        raise PredicitiveMaintainanceException(e, sys)

if __name__ == '__main__':
    main()