from .transformers_backend import TransformersBackend
from config import Config, BackendType
from .base import Backend


BACKENDS: dict[BackendType, Backend] = {
    'transformers': TransformersBackend
}


def backend_from_config(config: Config):
    backend = BACKENDS[config.backend]
    return backend(config.model_name, config.precision, config.max_new_tokens, config.device)