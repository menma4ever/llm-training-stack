"""Parameter-Efficient Fine-Tuning (LoRA/PEFT) pipeline."""

from llm_training_stack.config.schema import PeftConfig
from llm_training_stack.pipelines.sft import SFTPipeline


class LoRAPipeline(SFTPipeline):
    """Integrates LoRA rank-decomposition adapters into attention and projection weights.

    Maintains backward compatibility with task_type='lora' while sharing the orthogonal
    PEFT implementation from BaseTrainingPipeline.
    """

    def __init__(self, config):
        if config.peft is None:
            config.peft = PeftConfig()
        super().__init__(config)

    def setup_model_and_tokenizer(self) -> None:
        super().setup_model_and_tokenizer()
