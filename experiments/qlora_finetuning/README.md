# QLoRA Medical Instruction Fine-Tuning

Parameter-efficient fine-tuning experiment using Qwen2.5-1.5B-Instruct, 4-bit NF4 quantization, and LoRA adapters.

The notebook covers data loading and normalization, sequence-length diagnostics, baseline generation, supervised fine-tuning, adapter packaging, ROUGE-L and LLM-as-a-Judge evaluation, safety red-team prompts, and optional merged-weight export.

```bash
pip install -r ../../requirements/finetuning.txt
cp ../../.env.example .env
jupyter lab qlora_medical_finetuning.ipynb
```

A CUDA GPU is required for the pinned bitsandbytes 4-bit training path. Use Python 3.10 to 3.12. For GPU training, install the PyTorch build that matches the CUDA runtime on the target machine before installing the remaining dependencies. All saved outputs were intentionally cleared before publication. Run the notebook to regenerate outputs for your own environment.

## Audit note

Evaluation decodes only newly generated token IDs rather than slicing decoded prompt text. This avoids special-token offset errors that can make completions appear truncated. The earlier metric run was invalidated by this bug and is not presented as a final benchmark. Regenerate metrics with the corrected notebook before reporting performance.

## Current behavior

- Failed dataset downloads stop execution instead of substituting a repeated synthetic dataset.
- Blank examples and duplicates are removed before splitting. Identical prompts remain in one split.
- Token diagnostics use the same native chat format as training. Prompt-only truncated examples are rejected, and loss is computed on assistant tokens.
- Post-training inference reuses the trained model and sets evaluation mode.
- `CONFIG["push_to_hub"]` must be true before `HF_REPO_ID` can enable publishing. Inference always evaluates the current local training result.
- ROUGE-only runs still collect low-quality examples. Judge scores must be integers from 1 to 5 and receive the full instruction plus input context.
- The frozen base model and adapter are evaluated on identical prompts. Predictions and settings are saved in `outputs/evaluation.json`.
- Set `export_merged` and `export_gguf` explicitly to run the optional export stages. Subprocess failures propagate.

The recorded parameter counts describe the earlier run. Deduplication, response masking, and truncation filtering change the training protocol, so new quality metrics must be regenerated.
