import sys
from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import DatabaseConfig
from mysql.connector import connect

class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
    
    def connect(self):
        try:
            self.connection = connect(
                host = self.config.host,
                user = self.config.username,
                password = self.config.password,
                port = self.config.port,
                database = self.config.database
            )

            logging.info('Connected to the Database')
            
            return self.connection
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def execute(self, query: str, params: tuple = None):
        try:
            if self.connection is None:
                self.connect()
            
            cursor = self.connection.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def execute_many(self, query : str, params : list):
        try:
            if self.connection is None:
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.executemany(query, params)

            return cursor
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
        
    def fetch_one(self, query : str, params : tuple = None):
        try:
            cursor = self.execute(query=query, params=params)
            columns = [column[0] for column in cursor.description]
            row = cursor.fetchone()

            if row is None:
                return None
            
            return dict(zip(columns, row))

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def fetch_all(self, query : str, params : tuple = None):
        cursor = self.execute(query=query, params=params)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()

        return columns, rows
    
    def commit(self):
        try:
            if self.connection:
                self.connection.commit()
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def rollback(self):
        try:
            if self.connection:
                self.connection.rollback()
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def close(self):
        try:
            if self.connection:
                self.connection.close()
                self.connection = None

                logging.info("Database connection closed")
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)