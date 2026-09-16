# LoRA and QLoRA

LoRA freezes the base model and trains small low-rank adapter matrices. Only a fraction of parameters (often under 1%) are trainable.

QLoRA combines a 4-bit quantized base with those adapters so you can fine-tune on a smaller GPU. The artifact you ship is a tiny adapter file, not a full copy of the weights.

Lab 4 trains a classroom adapter on ten deployment Q&A pairs.
