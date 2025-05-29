# merge_lora_CPU# merge_lora_CPU
Simple LoRA merging with a base model, primarily designed for CPU but supports CUDA.

This script allows you to merge a PEFT (Parameter-Efficient Fine-Tuning) LoRA adapter with a base Hugging Face model. It can also be used to simply resave a base model, potentially changing its format (e.g., to SafeTensors) or data type.

## Prerequisites

Make sure you have the necessary Python libraries installed. If you are using an environment like Oobabooga's text-generation-webui, these libraries might already be present.

### Setting up a Virtual Environment (Recommended)

1.  **Create a virtual environment** (e.g., named `.venv`):
    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**
    *   **On Windows (Command Prompt or PowerShell):**
        ```bash
        .venv\Scripts\activate
        ```
    *   **On macOS/Linux (bash/zsh):**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install required libraries:**
    ```bash
    pip install torch transformers peft sentencepiece accelerate
    ```
    *   **Note on PyTorch for GPU:** If you intend to use a GPU, it's highly recommended to install PyTorch by following the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) to ensure compatibility with your CUDA version.

4.  **When you're done using the environment, deactivate it:**
    ```bash
    deactivate
    ```

## Usage

The script is run from the command line. You can see all available options by running:


python merge_tool.py --help

usage: merge_tool.py --model_path MODEL_PATH --output_path OUTPUT_PATH
                     [--lora_path LORA_PATH]
                     [--lora_checkpoint LORA_CHECKPOINT]
                     [--device {cpu,cuda,auto}] [--gpu_memory GPU_MEMORY]
                     [--cpu_memory CPU_MEMORY] [--safetensors]
                     [--no_safetensors]
                     [--dtype {float16,bfloat16,float32}] [-h]

Merge a PEFT LoRA with a base Hugging Face model.

options:
  --model_path MODEL_PATH
                        Path to the base Hugging Face model directory.
  --output_path OUTPUT_PATH
                        Path to save the merged model.
  --lora_path LORA_PATH
                        Path to the PEFT LoRA directory. (Optional, use
                        'None' or omit to skip LoRA merge)
  --lora_checkpoint LORA_CHECKPOINT
                        Name of the LoRA checkpoint subfolder (e.g.,
                        'checkpoint-1000'). 'Final' or empty means use the
                        main LoRA path. Default: 'Final'.
  --device {cpu,cuda,auto}
                        Device to use ('cpu', 'cuda', 'auto'). 'auto' will
                        use CUDA if available. Default: 'cpu'.
  --gpu_memory GPU_MEMORY
                        Maximum GPU memory (GiB) per device. 0 for no limit.
                        Default: 0.
  --cpu_memory CPU_MEMORY
                        Maximum CPU memory (GiB). 0 for no limit. Default: 0.
  --safetensors         Save with SafeTensors. (Default)
  --no_safetensors      Do not save with SafeTensors (use PyTorch .bin
                        files).
  --dtype {float16,bfloat16,float32}
                        Torch dtype for loading the model (float16, bfloat16,
                        float32). Default: float16.
  -h, --help            show this help message and exit

How to use it:

Run from your terminal:

##Example 1: Merge with LoRA on CPU (default)

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter" \
    --output_path "/path/to/your/merged_model_output_cpu"

    
## Example 2: Merge with a specific LoRA checkpoint on CPU

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter_parent_folder" \
    --lora_checkpoint "checkpoint-500" \
    --output_path "/path/to/your/merged_model_checkpoint_output"

    
## Example 3: "Merge" (resave) base model only, on CPU

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --output_path "/path/to/your/resaved_base_model"

    

(You can also explicitly pass --lora_path None)

## Example 4: Merge on CUDA (if available), limit GPU memory to 20GiB, CPU to 30GiB

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter" \
    --output_path "/path/to/your/merged_model_output_gpu" \
    --device cuda \
    --gpu_memory 20 \
    --cpu_memory 30

    


## Example 5: Merge using bfloat16 and save as .bin (not safetensors)

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter" \
    --output_path "/path/to/your/merged_model_bf16_bin" \
    --dtype bfloat16 \
    --no_safetensors

    


