import logging
import streamlit as st
from datetime import datetime
import os

class StreamlitLogger:
    def __init__(self, log_level=logging.INFO):
        self.logs = []
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.log_dir}/app_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message):
        self.logger.info(message)
        self.logs.append(f"INFO: {message}")
    
    def debug(self, message):
        self.logger.debug(message)
        self.logs.append(f"DEBUG: {message}")
    
    def warning(self, message):
        self.logger.warning(message)
        self.logs.append(f"WARNING: {message}")
    
    def error(self, message):
        self.logger.error(message)
        self.logs.append(f"ERROR: {message}")
    
    def get_logs(self):
        return "\n".join(self.logs)

class StreamlitLogger:
    def __init__(self):
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.log_file = os.path.join(self.log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def info(self, message: str):
        """Log info message and display in Streamlit"""
        self.logger.info(message)
        st.info(message)
        
    def warning(self, message: str):
        """Log warning message and display in Streamlit"""
        self.logger.warning(message)
        st.warning(message)
        
    def error(self, message: str):
        """Log error message and display in Streamlit"""
        self.logger.error(message)
        st.error(message)
        
    def debug(self, message: str):
        """Log debug message (file only)"""
        self.logger.debug(message)
        
    def get_logs(self) -> str:
        """Read and return all logs from file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return f.read()
        return "No logs available"
