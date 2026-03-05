# TinyAgent: Efficient Tool-Learning for Small Language Models 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![LLM](https://img.shields.io/badge/Llama_3-3B-orange)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green)

##  Project Overview
While Large Language Models (LLMs) demonstrate impressive generalization, their deployment on edge devices remains a challenge due to computational costs. This project investigates the plasticity of Small Language Models (SLMs) in complex agentic tasks. 

We engineered an end-to-end **Supervised Fine-Tuning (SFT)** pipeline to adapt a `Llama-3-3B` model into a "TinyAgent" capable of mastering function-calling and intent recognition.

##  Methodology & Tech Stack
* **Base Model:** `unsloth/Llama-3.2-3B-Instruct`
* **Parameter-Efficient Fine-Tuning (PEFT):** Leveraged **QLoRA (4-bit quantization)** via the **Unsloth** framework. This setup enables extremely fast training dynamics and reduces memory footprint, making it fully executable on consumer-grade hardware (NVIDIA T4 16GB).
* **Data Synthesis Pipeline:** Overcame data scarcity by implementing a 'Teacher-Student' distillation approach, synthesizing domain-specific instruction data with **Chain-of-Thought (CoT)** reasoning traces.

##  Tool-Calling Dataset Structure
To train the agent to "think" before acting, the dataset is structured to force the model to output a reasoning trace (`Thought`) before executing an API call (`Action`).

**Conceptual Breakdown:**
1. **Instruction:** The user's natural language request.
2. **Thought:** The internal reasoning step to decide which tool to use and extract arguments.
3. **Action:** The formatted JSON string that triggers the external API.

**Actual JSON Implementation Format:**

```json
{
  "instruction": "Calculate the total price if I buy 25 apples at $4 each.",
  "output": "Thought: I need to multiply the number of apples by the price per apple to get the total cost. \nAction: <tool_call> {'name': 'multiply', 'args': {'a': 25, 'b': 4}}"
}
```

##  How to Run

**1. Clone the repository:**

```bash
git clone https://github.com/YunqiWang1/TinyAgent-Llama3-FineTuning.git
cd TinyAgent-Llama3-FineTuning
```

**2. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the Training Script:**

If you are in a local terminal (Linux/Windows with GPU):
```bash
python sft_trainer.py
```

If you are running in a Google Colab notebook cell:
```python
!python sft_trainer.py
```

##  Expected Outcomes & Metrics
* **Enhanced Intent Recognition:** The QLoRA-adapted model achieved a baseline **57% accuracy** on complex, multi-variable tool-calling and intent extraction tasks, significantly outperforming the un-tuned base model.
* **CoT Emergence:** The agent successfully adopts a step-by-step reasoning process before selecting a tool, drastically reducing hallucinated arguments.
* **Dataset Transparency:** Please refer to the `dataset_sample/train_sample.jsonl` file in this repository to inspect the exact synthetic instruction formats used during the SFT phase.

##  Future Work
* **AI Alignment:** Implement **Direct Preference Optimization (DPO)** to steer the agent's behavioral tone and ensure safe tool usage (See my subsequent project: `Llama3-DPO-Alignment`).
* **Multi-Turn Execution:** Expand the dataset to support multi-step tool execution and error-handling loops.

##  Citation & License
This project is open-sourced under the MIT License.
