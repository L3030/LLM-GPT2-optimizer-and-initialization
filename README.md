# gpt2-lora-practice
This is a practice project for lora gpt2, with performance evaluation with different optimizers and schedulars.

## Details
We lora the last two layers of gpt2, including the attn layer as well as the fnn layer, on dataset used in [official LoRA example](https://github.com/microsoft/LoRA). However, it meets some trouble when fine-tune LN layer. 

We use AdamW, CAME, adafactor as optimizers; LambdaLR, ExponentialLR, CosineAnnealingLR and AdafactorSchedule as lr scheduler. Specifically, for AdamW and CAME, we found out that LambdaLR and CosineAnnealingLR both performs good, the latter is slightly better. For adafactor, it only works when using AdafactorSchedule but the convergence performance is still poor, and all these optimizers fails by using ExponentiallR. To understand why, we further try different initialization schemes, by replacing kaiming_uniform_ by kaiming_normal_, but the situation did not improve.

It seems the existing lora method can not applied into LN layer, may consider using tuning method in "Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning".

## Observations
The illustration of different methods can be found in "./save/attention/attention" and "./save/fnn/fnn".

The lora performance by using AdamW and CAME is stable and good. adafactor may needs more tuned for working normally, since the mean of lora matrix shows a monotonically changing trend.

For AdamW, the last layer's mean decrease fast at first, similarly for last second layer, implying a larger initial learning rate with dacay for lora.

For CAME, the last layer's mean changes dramatically, may sugggesting a smaller learning rate; for the last second layer's mean of lora matrix, the value of lora matrix continues to be larger than zero, may implying an initialization scheme with larger value.

For adafactor, we try different lora rank and different initialization schemes, but it always fails to converge in this setting.
