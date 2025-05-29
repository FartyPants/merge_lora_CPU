import argparse
import os
import shutil
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer # Using Auto classes for more generality

# FPHAM https://github.com/FartyPants/merge_lora_CPU

def process_merge(
    base_model_path_str: str,
    peft_model_path_str: str,
    lora_checkpoint_name: str, # Name of the checkpoint subfolder, or "Final" / None
    output_path_str: str,
    device_option: str = "cpu", # "cpu", "cuda", "auto"
    max_gpu_memory_gb: int = 0, # In GB
    max_cpu_memory_gb: int = 0, # In GB
    use_safetensors: bool = True,
    torch_dtype_str: str = "float16" # "float16", "bfloat16", "float32"
):
    print("--- Starting Model Merge Process ---")

    base_model_path = Path(base_model_path_str)
    output_path = Path(output_path_str)

    if not base_model_path.exists():
        print(f"ERROR: Base model path does not exist: {base_model_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Determine torch_dtype
    if torch_dtype_str == "float16":
        dtype = torch.float16
    elif torch_dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype_str == "float32":
        dtype = torch.float32
    else:
        print(f"Warning: Unknown torch_dtype '{torch_dtype_str}'. Defaulting to float16.")
        dtype = torch.float16
    print(f"Using dtype: {dtype}")

    # Configure max_memory
    max_memory = {}
    if device_option != "cpu" and max_gpu_memory_gb > 0 and torch.cuda.is_available():
        # Using GiB for transformers compatibility if that was the original intent,
        # but GB is more common for user input. Let's assume GiB for consistency with original.
        max_memory.update({i: f"{max_gpu_memory_gb}GiB" for i in range(torch.cuda.device_count())})
        print(f"Max GPU memory per device: {max_gpu_memory_gb}GiB")
    
    if max_cpu_memory_gb > 0:
        max_memory["cpu"] = f"{max_cpu_memory_gb}GiB"
        print(f"Max CPU memory: {max_cpu_memory_gb}GiB")
    
    if not max_memory: # If both are 0, max_memory remains empty, transformers will manage.
        max_memory = None # Pass None to from_pretrained

    # Determine device_map
    device_map_arg = "auto" # Default for CUDA or MPS if available
    if device_option == "cpu":
        device_map_arg = {"": "cpu"}
        print("Device map: Forcing CPU")
    elif device_option == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA specified but not available. Falling back to CPU.")
            device_map_arg = {"": "cpu"}
        else:
            print("Device map: Auto (CUDA preferred)")
    else: # auto
        print(f"Device map: Auto (selected: {device_option})")


    print(f"Loading base model: {base_model_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False, # Merging requires full precision model
            torch_dtype=dtype,
            device_map=device_map_arg,
            trust_remote_code=True, # Common for custom models
            max_memory=max_memory,
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    if peft_model_path_str and peft_model_path_str.lower() != "none":
        peft_model_actual_path = Path(peft_model_path_str)
        if lora_checkpoint_name and lora_checkpoint_name.lower() not in ['final', '']:
            peft_model_actual_path = peft_model_actual_path / lora_checkpoint_name
        
        if not peft_model_actual_path.exists():
            print(f"ERROR: LoRA path does not exist: {peft_model_actual_path}")
            # Option: proceed to save base model only, or exit. Let's exit.
            return

        print(f"Loading PEFT LoRA: {peft_model_actual_path}")
        try:
            lora_model = PeftModel.from_pretrained(
                base_model,
                peft_model_actual_path,
                device_map=device_map_arg, # Can try to match base_model's device_map
                torch_dtype=dtype,
                max_memory=max_memory, # In case LoRA itself is large
            )
        except Exception as e:
            print(f"Error loading PEFT model: {e}")
            return

        print("Running merge_and_unload...")
        try:
            base_model = lora_model.merge_and_unload() # Updates base_model in place
            base_model.train(False) # Set to eval mode
            print("Merge successful.")
        except Exception as e:
            print(f"Error during merge_and_unload: {e}")
            return
    else:
        print("No LoRA specified or 'None'. Proceeding to save base model as is.")

    print(f"Saving merged model to {output_path} (safetensors: {use_safetensors})")
    try:
        # The `base_model` variable now holds the (potentially merged) model
        base_model.save_pretrained(output_path, safe_serialization=use_safetensors)
    except Exception as e:
        print(f"Error saving model: {e}")
        return

    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
        # Continue, as model might have saved

    # Create a _merge.txt file for reference
    merge_info_path = output_path / "_merge_info.txt"
    with open(merge_info_path, 'w') as f:
        f.write(f"Base Model: {base_model_path.name}\n")
        if peft_model_path_str and peft_model_path_str.lower() != "none":
            f.write(f"LoRA: {Path(peft_model_path_str).name}\n")
            if lora_checkpoint_name and lora_checkpoint_name.lower() not in ['final', '']:
                f.write(f"LoRA Checkpoint: {lora_checkpoint_name}\n")
        f.write(f"Output Format: {'SafeTensors' if use_safetensors else 'PyTorch Binaries'}\n")
        f.write(f"Torch Dtype: {torch_dtype_str}\n")

    print(f"Merge information saved to {merge_info_path}")
    print("--- Model Merge Process Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA with a base Hugging Face model.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the base Hugging Face model directory."
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to the PEFT LoRA directory. (Optional, use 'None' or omit to skip LoRA merge)"
    )
    parser.add_argument(
        "--lora_checkpoint", type=str, default="Final", help="Name of the LoRA checkpoint subfolder (e.g., 'checkpoint-1000'). 'Final' or empty means use the main LoRA path. Default: 'Final'."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the merged model."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"],
        help="Device to use ('cpu', 'cuda', 'auto'). 'auto' will use CUDA if available. Default: 'cpu'."
    )
    parser.add_argument(
        "--gpu_memory", type=int, default=0, help="Maximum GPU memory (GiB) per device. 0 for no limit. Default: 0."
    )
    parser.add_argument(
        "--cpu_memory", type=int, default=0, help="Maximum CPU memory (GiB). 0 for no limit. Default: 0."
    )
    parser.add_argument(
        "--safetensors", action="store_true", default=True, help="Save with SafeTensors. (Default)"
    )
    parser.add_argument(
        "--no_safetensors", action="store_false", dest="safetensors", help="Do not save with SafeTensors (use PyTorch .bin files)."
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for loading the model (float16, bfloat16, float32). Default: float16."
    )

    print("https://github.com/FartyPants/merge_lora_CPU")
    print("********************************************")

    args = parser.parse_args()

    '''
    python merge_cpu.py \
    --model_path "/path/to/your/base_model_hf" \
    --lora_path "/path/to/your/lora_adapter" \
    --output_path "/path/to/your/merged_model_output_cpu"
    '''

    process_merge(
        base_model_path_str=args.model_path,
        peft_model_path_str=args.lora_path,
        lora_checkpoint_name=args.lora_checkpoint,
        output_path_str=args.output_path,
        device_option=args.device,
        max_gpu_memory_gb=args.gpu_memory,
        max_cpu_memory_gb=args.cpu_memory,
        use_safetensors=args.safetensors,
        torch_dtype_str=args.dtype
    )