import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

class TinyAgentTrainer:
    def __init__(self, model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=2048):
        self.max_seq_length = max_seq_length
        print("[INFO] Loading base model and tokenizer (4-bit quantization)...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )
        self._apply_lora()

    def _apply_lora(self):
        print("[INFO] Applying LoRA adapters for Parameter-Efficient Fine-Tuning...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    def prepare_data(self):
        print("[INFO] Preparing synthetic tool-calling dataset...")
        # Example synthetic data for intent recognition and tool calling
        data = [
            {"instruction": "Calculate 25 * 4", "output": "Thought: I need to multiply two numbers. Action: <tool_call> {'name': 'multiply', 'args': {'a': 25, 'b': 4}}"},
            {"instruction": "What is 100 + 50?", "output": "Thought: I need to add two numbers. Action: <tool_call> {'name': 'add', 'args': {'a': 100, 'b': 50}}"}
        ]
        return Dataset.from_list(data)

    def train(self):
        dataset = self.prepare_data()
        print("[INFO] Initializing SFT Trainer...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="output",
            max_seq_length=self.max_seq_length,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=120,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                output_dir="sft_outputs",
            ),
        )
        print("[INFO] Starting SFT training loop...")
        trainer.train()
        print("[INFO] Training complete! Model saved to ./sft_outputs")

if __name__ == "__main__":
    agent = TinyAgentTrainer()
    agent.train()