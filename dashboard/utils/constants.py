import os
from dotenv import load_dotenv
load_dotenv()

API_HOST = os.getenv('API_HOST')
API_PORT = os.getenv('API_PORT')

# Backend Config

BASE_API_URL = f'http://{API_HOST}:{API_PORT}'
WEBSOCKET_URL = f'ws://{API_HOST}:{API_PORT}/ws/'

# API Endpoints

ROOT_ENDPOINT = '/'
HEALTH_ENDPOINT = '/health'
ENGINES_ENDPOINT = '/engines'
STREAM_PREDICTION_ENDPOINT = '/predict/stream'
BATCH_PREDICTION_ENDPOINT = '/predict/batch'

# Dashboard Config

AUTO_REFRESH_INTERVAL = 5   # seconds
PAGE_SIZE = 20
DEFAULT_ENGINE_ID = 1

# Alert Threshold

FAILURE_IMMINENT_RUL = 10
CRITICAL_RUL = 30
WARNING_RUL = 60
CRITICAL_FAILURE_PROBABILTY = 0.80
WARNING_FAILURE_PROBABILITY = 0.50

# Alert Labels

SAFE = 'SAFE'
WARNING = 'WARNING'
CRITICAL = 'CRITICAL'
FAILURE_IMMINENT = 'FAILURE_IMMINENT'

# Status Colors

STATUS_COLORS = {
    SAFE : '#28A745',
    WARNING : '#FFC107',
    CRITICAL : '#FF6B35',
    FAILURE_IMMINENT : '#DC3545'
}

# Engine Table Columns

ENGINE_TABLE_COLUMNS = [
    'engine_id',
    'cycle',
    'predicted_rul',
    'failure_probability',
    'alert_level'
]

# Metric Card Labels

METRIC_TOTAL = 'Total Engines'
METRIC_HEALTHY = 'Healthy'
METRIC_WARNING = 'Warning'
METRIC_CRITICAL = 'Critical'
METRIC_AVG_RUL = 'Average RUL'

# Plot Config

CHART_HEIGHT = 400
RUL_COLOR = '#1F77B4'
FAILURE_COLOR = '#D62728'
HEALTH_COLOR = '#2CA02C'

# Upload Config

SUPPORTED_FILE_TYPE = ['csv']
MAX_UPLOAD_SIZE_MB = 25