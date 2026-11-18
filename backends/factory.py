from .transformers_backend import TransformersBackend
from .vllm_backend import vLLMBackend
from config import InferenceConfig, BackendType
from .base import Backend


BACKENDS: dict[BackendType, type[Backend]] = {
    'transformers': TransformersBackend,
    'vllm': vLLMBackend
}


def backend_from_config(config: InferenceConfig):
    backend = BACKENDS[config.backend]
    return backend(config.model_name, config.precision, config.device, config.compile, config.batch_size)