import sys
from predictivesystem.logging import logger

class PredictiveMaintenanceException(Exception):
    def __init__(self, error_msg, error_details : sys):
        self.error_msg = error_msg
        _, _, exc_tb = error_details.exc_info()

        self.line_no = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return f"Error occured in script name [{self.file_name}] line number [{self.line_no}] error message [{self.error_msg}]"