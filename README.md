# merge_lora_CPU
Simple LORA merging with model on CPU

make sure you have transformers, peft and all that installed.

For example run cmd_windows.bat in your oobabooga WebUi directory and run it from there in venv as it has required libraries. 


How to use it:

Run from your terminal:

Example 1: Merge with LoRA on CPU (default)

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter" \
    --output_path "/path/to/your/merged_model_output_cpu"

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Example 2: Merge with a specific LoRA checkpoint on CPU

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter_parent_folder" \
    --lora_checkpoint "checkpoint-500" \
    --output_path "/path/to/your/merged_model_checkpoint_output"

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Example 3: "Merge" (resave) base model only, on CPU

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --output_path "/path/to/your/resaved_base_model"

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

(You can also explicitly pass --lora_path None)

Example 4: Merge on CUDA (if available), limit GPU memory to 20GiB, CPU to 30GiB

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter" \
    --output_path "/path/to/your/merged_model_output_gpu" \
    --device cuda \
    --gpu_memory 20 \
    --cpu_memory 30

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Example 5: Merge using bfloat16 and save as .bin (not safetensors)

      
python merge_tool.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter" \
    --output_path "/path/to/your/merged_model_bf16_bin" \
    --dtype bfloat16 \
    --no_safetensors

    


