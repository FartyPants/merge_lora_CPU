import argparse
from pathlib import Path
import torch
from peft import PeftModel, tuners
from transformers import AutoModelForCausalLM, AutoTokenizer # Using Auto classes for more generality
import json
from safetensors.torch import save_model


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
    torch_dtype_str: str = "float16", # "float16", "bfloat16", "float32"
    alpha_value: int = 0,  # Default to 0, no direct change
    alpha_perc: int = 0 # Default to 0, no percentage change
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
        current_device = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"
        print(f"Device map: Auto (will attempt to use best available: {current_device})")

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
    


    # Use a dictionary to store the short module name and a full path example
    module_examples = {}

    # Iterate through all named modules in the model
    for name, module in base_model.named_modules():
        # We are looking for the linear layers within attention and MLP blocks.
        if isinstance(module, torch.nn.Linear) and ("attn" in name or "mlp" in name):
            # The target module name is the last part of the full name string.
            # e.g., in 'language_model.model.layers.0.self_attn.q_proj', we want 'q_proj'
            module_type = name.split('.')[-1]
            
            # If we haven't seen this module type before, save it with its full path
            if module_type not in module_examples:
                module_examples[module_type] = name

    print("\n--- Potential LoRA Target Modules ---")
    print("Shows the short name (for 'target_modules') and a full path example (for debugging).")

    if not module_examples:
        print("\nNo potential targetable linear layers found in 'attn' or 'mlp' blocks.")
        return
     
    # Sort by the short module name for consistent, alphabetical output
    # The f-string formatting `{key:<12}` pads the key to 12 characters for alignment
    for key in sorted(module_examples.keys()):
        print(f"- {key:<12} (e.g., {module_examples[key]})")
    
    # Extract just the unique module types for the JSON example
    unique_target_modules = sorted(list(module_examples.keys()))

    #print("\n--- Example `adapter_config.json` ---")
    #print("You can use this list for the 'target_modules' parameter when training a LoRA.")
    #print(json.dumps({"target_modules": unique_target_modules}, indent=2))
    
    print("\n--- Analysis Complete ---")



    if peft_model_path_str and peft_model_path_str.lower() != "none":
        peft_model_actual_path = Path(peft_model_path_str)
        if lora_checkpoint_name and lora_checkpoint_name.lower() not in ['final', '']:
            peft_model_actual_path = peft_model_actual_path / lora_checkpoint_name
        
        if not peft_model_actual_path.exists():
            print(f"ERROR: LoRA path does not exist: {peft_model_actual_path}")
            # Option: proceed to save base model only, or exit. Let's exit.
            return
        
        adapter_config_path = peft_model_actual_path / "adapter_config.json"

        # --- Alpha value modification logic START ---
        # Only modify if alpha_value is a positive integer (not 0)

        original_lora_alpha = None

        if alpha_value > 0 or alpha_perc > 0:
            if adapter_config_path.exists():
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    
                    if "lora_alpha" in adapter_config:
                        original_lora_alpha = adapter_config["lora_alpha"]
                        print(f"Original lora_alpha: {original_lora_alpha}")

                        if alpha_value > 0: # --alpha takes precedence
                            print(f"Using --alpha value ({alpha_value})")
                            adapter_config["lora_alpha"] = alpha_value
                            print(f"Temporarily setting lora_alpha to: {alpha_value}")
                        elif alpha_perc > 0: # If --alpha_perc is set and --alpha is not
                            percent_alpha = float(alpha_perc) / 100.0
                            calculated_alpha = int(float(original_lora_alpha) * percent_alpha)
                            print(f"Calculating lora_alpha using --alpha_perc ({alpha_perc}% of {original_lora_alpha}): {calculated_alpha}")
                            adapter_config["lora_alpha"] = calculated_alpha
                        
                        # --- WRITE BACK ---
                        with open(adapter_config_path, 'w') as f:
                            json.dump(adapter_config, f, indent=2)
                        print(f"Successfully modified lora_alpha in {adapter_config_path}")
                    else:
                        print(f"Warning: 'lora_alpha' key not found in {adapter_config_path}. Cannot modify.")
                        original_lora_alpha = None # Ensure we don't try to restore
                        adapter_config_path = None
                except Exception as e:
                    print(f"Error modifying {adapter_config_path}: {e}")
                    original_lora_alpha = None # Ensure we don't try to restore
                    adapter_config_path = None # Prevent restoration attempts
            else:
                print(f"Warning: {adapter_config_path} not found. Cannot modify lora_alpha.")
                original_lora_alpha = None # Ensure we don't try to restore
                adapter_config_path = None # Prevent restoration attempts

        # --- Alpha value modification logic END ---        

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

        print("\n--- Analyzing Injected PEFT Adapter Layers ---")
        print("This shows the layers that were successfully replaced by PEFT's LoRA modules.")
        
        adapted_modules = {}
        for name, module in lora_model.named_modules():
            # Check for the specific LoRA layer types from the PEFT library
            if isinstance(module, tuners.lora.Linear):
                module_type = name.split('.')[-1]
                if module_type not in adapted_modules:
                    adapted_modules[module_type] = name
        
        if not adapted_modules:
            print("WARNING: No PEFT LoRA layers were found injected in the model.")
        else:
            print("Found the following adapted modules:")
            for key in sorted(adapted_modules.keys()):
                print(f"- {key:<12} (e.g., {adapted_modules[key]})")
        print("--- Adapter Analysis Complete ---\n")
 


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

     
    # --- This is the NEW, corrected block ---
    print(f"Saving merged model to {output_path} (safetensors: {use_safetensors})")
    try:
        if use_safetensors:
            try:
                # ATTEMPT 1: Try the simple, standard method for saving the WHOLE model.
                print("Attempting standard save_pretrained with safetensors...")
                base_model.save_pretrained(output_path, safe_serialization=True)
                print("Standard save successful.")
            except RuntimeError as e:
                print("\nStandard save failed due to tied weights. Using fallback method.")
                print("Saving config.json...")
                base_model.config.save_pretrained(output_path)
                print("Saving model.safetensors...")
                save_model(base_model, output_path / "model.safetensors", metadata={'format': 'pt'})
                print("Fallback save for tied weights finished.")

 
                # 3. For compatibility, we might need to create an index file.
                # Seems to only muddy things
                # This tells transformers how to map the tensors in the file.
                #index_data = {
                #    "metadata": {},
                #    "weight_map": {
                #        tensor_name: "model.safetensors" 
                #        for tensor_name in base_model.state_dict().keys()
                #    }
                #}
                #index_file_path = output_path / "model.safetensors.index.json"
                #with open(index_file_path, "w", encoding="utf-8") as f:
                #    json.dump(index_data, f, indent=2)

        else:
            # If not using safetensors, the old method for .bin files is fine.
            print("Saving with standard PyTorch .bin files.")
            base_model.save_pretrained(output_path, safe_serialization=False)
            
    except Exception as e:
        print(f"Error saving model: {e}")
        # It's helpful to print the full traceback for debugging
        import traceback
        traceback.print_exc()
        return

    

    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
        # Continue, as model might have saved

    actual_vocab_size = len(tokenizer)
    print(f"The ACTUAL vocabulary size is: {actual_vocab_size}")
    

    # Create a _merge.txt file for reference
    merge_info_path = output_path / "_merge_info.txt"
    with open(merge_info_path, 'w') as f:
        f.write(f"Base Model: {base_model_path.name}\n")
        if peft_model_path_str and peft_model_path_str.lower() != "none":
            f.write(f"LoRA: {Path(peft_model_path_str).name}\n")
            if lora_checkpoint_name and lora_checkpoint_name.lower() not in ['final', '']:
                f.write(f"LoRA Checkpoint: {lora_checkpoint_name}\n")
            else:
                 f.write(f"LoRA Checkpoint: None \n")
        else:
            f.write(f"LoRA: None (Base model saved as is)\n")

        f.write(f"Output Format: {'SafeTensors' if use_safetensors else 'PyTorch Binaries'}\n")
        f.write(f"Torch Dtype: {torch_dtype_str}\n")
        f.write(f"Device Option: {device_option}\n")

    print(f"Merge information saved to {merge_info_path}")

    # --- Alpha value restoration logic (guaranteed to run) START ---
    # Only restore if it was actually changed (original_lora_alpha is not None AND alpha_value > 0)
    if original_lora_alpha is not None and alpha_value > 0 and adapter_config_path and adapter_config_path.exists():
        print(f"Restoring lora_alpha in {adapter_config_path} to original value: {original_lora_alpha}")
        try:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_config["lora_alpha"] = original_lora_alpha
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f, indent=2)
            print("lora_alpha restored successfully.")
        except Exception as e:
            print(f"Error restoring lora_alpha in {adapter_config_path}: {e}")
    # --- Alpha value restoration logic END ---

    print("--- Model Merge Process Finished ---")

if __name__ == "__main__":
    print("FPHAM LoRA Merge Script (Enhanced)")
    print("Original: https://github.com/FartyPants/merge_lora_CPU")
    print("********************************************")

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
        "--output_path", type=str, default=None, help="Path to save the merged model. If not specified and --lora_path is given, defaults to '--lora_path/merge'"
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
    parser.add_argument(
        "--alpha", type=int, default=0, # Changed default to 0
        help="Optional: Value to temporarily set lora_alpha in adapter_config.json before merging. The original value will be restored after the merge "
             "If 0 (default), no change is made. "
             "if a positive value was used. Only applies if --lora_path is used."
    )

    parser.add_argument(
        "--alpha_perc", type=int, default=0,
        help="Optional: Calculate lora_alpha as a percentage of the original lora_alpha. The original value will be restored after the merge "
             "E.g., 50 for 50%. If 0 (default), no change. --alpha takes precedence. "
             "Only applies if --lora_path is used."
    )

    args = parser.parse_args()

    output_path_str_to_use = args.output_path

    if output_path_str_to_use is None:
        if args.lora_path and args.lora_path.lower() != "none":
            # Default to lora_path / "merge" as requested
            default_output_dir = Path(args.lora_path) / "merge"
            output_path_str_to_use = str(default_output_dir)
            print(f"INFO: --output_path not specified. Defaulting to: {output_path_str_to_use}")
        else:
            # If output_path is not given AND lora_path is not given (or is "None"),
            # it's an error because we can't form the default.
            parser.error(
                "ERROR: --output_path must be specified if --lora_path is not provided "
                "(or is 'None'), as the default output path (lora_path/merge) cannot be determined."
            )

    '''
    Example Usage:
    1. Merge with default output path:
    python your_script_name.py \
        --model_path "/path/to/base_model_hf" \
        --lora_path "/path/to/lora_adapter" \
        # Output will be /path/to/lora_adapter/merge

    2. Merge with specified output path:
    python your_script_name.py \
        --model_path "/path/to/base_model_hf" \
        --lora_path "/path/to/lora_adapter" \
        --output_path "/custom/output/dir"

    3. Save base model (no LoRA merge) - output_path is required:
    python your_script_name.py \
        --model_path "/path/to/base_model_hf" \
        --output_path "/path/to/base_model_resaved" \
        --lora_path "None" 
    '''

    process_merge(
        base_model_path_str=args.model_path,
        peft_model_path_str=args.lora_path,
        lora_checkpoint_name=args.lora_checkpoint,
        output_path_str=output_path_str_to_use, # Use the determined path
        device_option=args.device,
        max_gpu_memory_gb=args.gpu_memory,
        max_cpu_memory_gb=args.cpu_memory,
        use_safetensors=args.safetensors,
        torch_dtype_str=args.dtype,
        alpha_value=args.alpha,
        alpha_perc=args.alpha_perc
    )