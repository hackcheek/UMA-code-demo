import logging
import os
from freewillai.core import run_task, connect, upload_model, upload_dataset

LOG_FORMAT = '[%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s] %(message)s'
logging.basicConfig(
    filename=os.getenv('LOG_FILE'),
    level=logging.DEBUG if os.getenv('DEBUG') else logging.INFO,
    format=LOG_FORMAT
)

__version__ = '0.1.0'
__author__ = 'FreeWillAI'
__license__ = 'MIT'

__all__ = ['run_task', 'connect', 'upload_dataset', 'upload_model']
__doc__ = """
TODO
"""
