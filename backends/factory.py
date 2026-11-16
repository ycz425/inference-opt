from .transformers_backend import TransformersBackend
from config import InferenceConfig, BackendType
from .base import Backend


BACKENDS: dict[BackendType, type[Backend]] = {
    'transformers': TransformersBackend
}


def backend_from_config(config: InferenceConfig):
    backend = BACKENDS[config.backend]
    return backend(config.model_name, config.precision, config.device, config.compile)