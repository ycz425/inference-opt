from .transformers_backend import TransformersBackend
from config import Config, BackendType
from .base import Backend


BACKENDS: dict[BackendType, type[Backend]] = {
    'transformers': TransformersBackend
}


def backend_from_config(config: Config):
    backend = BACKENDS[config.backend]
    return backend(config.model_name, config.precision, config.output_tokens, config.device, config.seed)