import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import torch
from transformers import pipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eval-hh", action="store_true", help="Run small HH-style evaluation after model is ready")
args = parser.parse_args()

# Verify bitsandbytes installation
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes successfully imported. Version: {bnb.__version__}")
except ImportError as e:
    print(f"Error importing bitsandbytes: {e}")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA is available! GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

# # Configure 8-bit quantization
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Format datasets
def format_hh(example):
    full_conversation = example['chosen']
    parts = full_conversation.split("\n\nAssistant: ")
    human_part = parts[0].replace("Human: ", "").strip()
    assistant_part = parts[1].strip()
    return {"text": f"<|user|>\n{human_part}\n<|assistant|>\n{assistant_part}"}

def format_tqa_mc(example):
    q = example["question"]
    choices = example["mc1_targets"]["choices"]
    labels = example["mc1_targets"]["labels"]
    choices_str = "\n".join(f"- {c}" for c in choices)
    correct_answer_index = labels.index(1)
    ans = choices[correct_answer_index]
    return {"text": f"<|user|>\nAnswer the following multiple-choice question truthfully.\n\nQuestion:\n{q}\n\nChoices:\n{choices_str}\n\n<|assistant|>\n{ans}"}

def format_tqa_gen(example):
    q = example["question"]
    ans = example["best_answer"]
    return {"text": f"<|user|>\n{q}\n<|assistant|>\n{ans}"}

# Tokenize
def tokenize(ex):
    return tokenizer(ex["text"], truncation=True, max_length=512, padding="max_length")

def load_finetuned_for_inference(save_dir):
    print("Loading fine-tuned model for inference from:", save_dir)
    tokenizer_local = AutoTokenizer.from_pretrained(save_dir)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Try 1: load base fully in memory (no offload)
    try:
        print("Attempting full in-memory load of base model (no offload)...")
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            low_cpu_mem_usage=False,   # allow full load into memory
            dtype=torch.float32,
        )
        model_local = PeftModel.from_pretrained(base, save_dir, is_trainable=False)
        model_local.eval()
        gen = pipeline("text-generation", model=model_local, tokenizer=tokenizer_local)
        return model_local, tokenizer_local, gen

    except Exception as e:
        # Any error during in-memory load → fallback to offload-mode.
        import traceback
        print("In-memory load failed. Falling back to offload-mode. Exception:", e)
        traceback.print_exc()


    # Try 2: recreate offload folder and let accelerate rebuilt index
    offload_folder = os.path.join(os.getcwd(), "offload")
    if os.path.exists(offload_folder):
        # clear stale offload index to force a fresh rebuild
        try:
            import shutil
            shutil.rmtree(offload_folder)
        except Exception:
            pass
    os.makedirs(offload_folder, exist_ok=True)

    print("Loading base with offload_folder = ", offload_folder)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        dtype=torch.float32,
        offload_folder=offload_folder,
    )

    # Load adapter
    model_local = PeftModel.from_pretrained(
        base, 
        save_dir, 
        is_trainable=False, 
        offload_dir=offload_folder,
    )

    model_local.eval()
    gen = pipeline("text-generation", model=model_local, tokenizer=tokenizer_local)
    return model_local, tokenizer_local, gen

def adapter_exists(dirpath):
    candidates = ["adapter_config.json", "adapter_model.bin", "pytorch_model.bin", "pytorch_adapter.bin"]
    return any(os.path.exists(os.path.join(dirpath, f)) for f in candidates)

def evaluate_text_dataset(dataset, text_column="text", max_samples=100, out_path=None, max_new_tokens=200):
    """
    Run generation on up to `max_samples` examples from `dataset`.
    `dataset` can be a `datasets.Dataset` or an iterable of strings.
    Saves results to out_path if provided.
    """
    # prepare iterator of (prompt, reference) pairs
    if isinstance(dataset, str):
        from datasets import load_dataset as _load
        ds = _load(dataset)
        it = ((ex.get(text_column, ""), None) for ex in ds)
    elif hasattr(dataset, "select") or hasattr(dataset, "column_names"):
        it = ((ex.get(text_column, ""), ex.get("text", None)) for ex in dataset) # keep reference if present
    else:
        # assume iterable of strings
        it == ((s, None) for s in dataset)

    results = []
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_return_sequences": 1,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "top_k": 50,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    for i, (prompt, ref) in enumerate(it):
        if i >= max_samples:
            break
        try:
            out = generator(prompt, **gen_kwargs)
            text = out[0].get("generated_text") or out[0].get("text") or ""
        except Exception as e:
            text = f"<generation error: {e}>"
        results.append({"index": i, "prompt": prompt, "gen": text, "ref": ref})
        # progress print
        if (i+1) % 10 == 0 or i == max_samples - 1:
            print(f"Evaluated {i+1}/{min(max_samples, (len(dataset) if hasattr(dataset, '__len__') else max_samples))} examples")
        
        if out_path:
            import json
            with open(out_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        return results

def generate_with_pipeline(prompt, max_new_tokens=200, sample=False):
    """
    Quick deterministic check by default (sample=False).
    Set sample=True to run stochastic sampling but with conservative defaults.
    """
    if not sample:
        gen_kwargs = {
            "max_new_tokens": min(64, max_new_tokens),
            "do_sample": False,
            "num_return_sequences": 1,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }
    else:
        gen_kwargs = {
            "max_new_tokens": min(128, max_new_tokens),
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,                    # smaller top_k speeds up sampling
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 1,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }

    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # Try fast pipeline call first
    try:
        out = generator(prompt, **gen_kwargs)
        print(out[0].get("generated_text") or out[0].get("text", "") or "")
        return
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("Pipeline generation failed or hung:", e)

    # Fallback: direct model.generate (deterministic, explicit device placement)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out_ids = model.generate(**inputs, max_new_tokens=min(64, max_new_tokens), do_sample=False)
        print(tokenizer.decode(out_ids[0], skip_special_tokens=True))
    except Exception as e:
        print("Direct generate fallback failed:", e)

# Train / load / inference switch
save_directory = "./finetuned-lora-current"
os.makedirs(save_directory, exist_ok=True)

# If a saved adapter exists, skip training and load for inference
if adapter_exists(save_directory):
    model, tokenizer, generator = load_finetuned_for_inference(save_directory)
else:
    # If GPU is available use 8-bit/bnb paths, otherwise force CPU/offload-friendly load
    if torch.cuda.is_available():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        # GPU path (keep quantization if CUDA and a compatible bitsandbytes)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        # CPU / Mac path: avoid allocating huge warmup buffers, load on CPU with low memory usage
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~1.1B params vs Gemma's 2B 
        offload_folder = os.path.join(os.getcwd(), "offload")
        os.makedirs(offload_folder, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cpu"},         # force CPU device map
            low_cpu_mem_usage=True,         # reduce memory spikes during load
            dtype=torch.float32,            # explicit dtype for CPU
            offload_folder=offload_folder,  # where large tensors may be offloaded
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        # Using a subset of the 7 projection layers: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules=["q_proj", "v_proj", "k_proj"],
    )

    model = get_peft_model(model, lora_config)

    # Ensure model is in train mode and only LoRA/PEFT params require gradients
    model.train()
    for n, p in model.named_parameters():
        if ("lora" in n) or ("peft" in n) or ("adapter" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Sanity check before creating Trainer / starting backward
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total}")
    assert trainable > 0, "No parameters require gradients — check that LoRA adapters were added correctly."

    # Load datasets
    hh = load_dataset("Anthropic/hh-rlhf", split="train")
    tqa_mc = load_dataset("truthful_qa", "multiple_choice", split="validation")
    tqa_gen = load_dataset("truthful_qa", "generation", split="validation")

    hh_f = hh.map(format_hh)
    tqa_mc_f = tqa_mc.map(format_tqa_mc)
    tqa_gen_f = tqa_gen.map(format_tqa_gen)

    tqa_combined = Dataset.from_dict({
        "text": list(tqa_mc_f["text"]) + list(tqa_gen_f["text"])
    })

    tokenized_hh = hh_f.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_hh_small = tokenized_hh.select(range(int(len(tokenized_hh) * 0.01)))
    tokenized_tqa = tqa_combined.map(tokenize, batched=True, remove_columns=["text"])

    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=save_directory,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),                     # disable bf16 on CPU
        gradient_checkpointing=torch.cuda.is_available(),   # only enable on GPU
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_hh_small,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save the PEFT/LoRA adapters + tokenizer
    print("Saving fine-tuned adapters and tokenizer to:", save_directory)
    model.save_pretrained(save_directory)       # PEFT model saves only adapters by default
    tokenizer.save_pretrained(save_directory)

    # Load back in eval mode (useful for running inference in the same script)
    model, tokenizer, generator = load_finetuned_for_inference(save_directory)

print("Model ready for inference.")

# Inference setup: use `generator` if created by load_finetuned_for_inference(), otherwise create it from in-memory model
if 'generator' not in globals():
    model.eval()
    # If model was loaded into memory and CUDA is available, you could pass device=0.
    # But when accelerate/offload handled device_map, do NOT pass device.
    use_device = torch.cuda.is_available() and not hasattr(model, "hf_device_map")
    gen_kwargs = {"model": model, "tokenizer": tokenizer}
    if use_device:
        gen = pipeline("text-generation", **gen_kwargs, device=0)
    else:
        gen = pipeline("text-generation", **gen_kwargs)
    generator = gen

# quick debug checks (prints raw pipeline output and token ids)
print("Running quick generator sanity checks...")
prompt_debug = "<|user|>\nWhy is it dangerous to give medical advice without training?\n<|assistant|>"
out_raw = generator(prompt_debug, max_new_tokens=20, do_sample=False)
print("raw pipeline output:", out_raw)
print("tokenizer.encode:", tokenizer.encode(prompt_debug)[:80])
# Direct generate test (bypass pipeline) — may require enough memory
try:
    inputs = tokenizer(prompt_debug, return_tensors="pt")
    if next(model.parameters()).device.type == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    else:
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=50)
    print("direct decode:", tokenizer.decode(out_ids[0], skip_special_tokens=True)[:400])
except Exception as e:
    print("direct generate failed:", e)

# - Evaluate the tokenized_tqa you prepared during training (only available if training ran)
try:
    if 'tokenized_tqa' in locals():
        print("Running quick eval on tokenized_tqa (10 examples)...")
        # if tokenized_tqa contains tokenized feature, evaluate using the original tqa_combined
        # use the un-tokenized tqa_combined if available
        if 'tqa_combined' in locals():
            _ = evaluate_text_dataset(tqa_combined, max_samples=10, out_path="./eval_tqa_outputs.jsonl")
        else:
            _ = evaluate_text_dataset(tokenized_tqa, max_samples=10, out_path="./eval_tqa_outputs.jsonl")
except Exception:
    pass

if args.eval_hh:
    from datasets import load_dataset as _load
    print("Running HH quick eval (20 examples)...")
    hh_eval = _load("Anthropic/hh-rlhf", split="validation").map(format_hh).select(range(20))
    evaluate_text_dataset(hh_eval, max_samples=20, out_path="./eval_hh_outputs.jsonl")

if 'generator' in globals():
    print("=== Test 1: HH-style ===")
    generate_with_pipeline("<|user|>\nWhy is it dangerous to give medical advice without training?\n<|assistant|>")

    print("\n=== Test 2: TruthfulQA-style ===")
    generate_with_pipeline("<|user|>\nWhat happens if you eat too many carrots?\n<|assistant|>")
else:
    print("Generator not available - skipping quick manual tests.")
