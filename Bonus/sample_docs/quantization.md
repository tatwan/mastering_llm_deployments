# Quantization

Quantization reduces model weight precision so a GPU can hold the model.

NF4 is a 4-bit format designed for normally distributed LLM weights. Compared with FP16, it cuts memory roughly 4× while keeping enough quality for many deployment tasks.

Lab 4 measures this on a T4: load the same Qwen2.5 in FP16 and NF4, then compare VRAM and tokens/sec.
