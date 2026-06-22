import mlflow

from config import settings
from services.blob_loader import BlobLoader
from utils.sequence_buffer import SequenceBuffer

async def initialize_backend(app):
    loader = BlobLoader()

    metadata = loader.load_metadata()

    model_dir = loader.download_model(metadata['model_path'])

    model = mlflow.pyfunc.load_model(str(model_dir))

    app.state.hybrid_model = model
    app.state.feature_scaler = model.unwrap_python_model().feature_scaler
    app.state.metadata = metadata
    app.state.sequence_buffer = SequenceBuffer(settings.SEQUENCE_LENGTH)