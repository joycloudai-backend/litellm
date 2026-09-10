from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import RezecyanImageGenerationConfig

__all__ = [
    "RezecyanImageGenerationConfig",
    "get_rezecyan_image_generation_config",
]


def get_rezecyan_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return RezecyanImageGenerationConfig()
