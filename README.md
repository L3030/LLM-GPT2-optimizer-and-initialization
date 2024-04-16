![Lora A lastlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/dbda2e23-5642-432f-a7b9-a7f81acfa58e)# gpt2-lora-practice
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

# Update 2024.4.16 Upload the weights_distribution rar file.
We provide the histograms of parameter distributions for Lora A and Lora B in the second-to-last layer and last layer, for both the attention (attn) and feedforward neural network (fnn) layers. The distribution shifts are compared for AdamW, CAME, and Adafactor.

With one batch*gradient_accumulation_steps, for every gradient_accumulation_step, we report the histograms of parameter distributions because we want to observe the changes in distributions as they converge rapidly. After that, we report the distribution every num_update_steps, since the distribution changes become small as the model tends to converge.

## AdamW
For example, in AdamW, Lora A of last layer starts with uniform distribution:
![Lora A lastlayer - Epoch 15_within a batch](https://github.com/L3030/gpt2-lora-practice/assets/74963049/4ed6ff8f-64ae-4807-80b4-0a8e20128e32), and after one entire num_update_steps, the Lora A parameter varys toward a Gaussian distribution with 0 as the mean:
![Lora A lastlayer - Epoch 1](https://github.com/L3030/gpt2-lora-practice/assets/74963049/265c0b38-d62c-4f27-971f-930eb0771001). As the number of training iterations increases, the model parameters eventually become distributed in a narrower Gaussian distribution:
![Lora A lastlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/697a9a47-a711-4a88-bc51-55f4077dfe29).

Similarly, in Adamw, for Lora B of the last layer, it starts with a zero distribution, and after one round's update, the parameter becomes concave distribution:![Lora B lastlayer - Epoch 47_within a batch](https://github.com/L3030/gpt2-lora-practice/assets/74963049/458865b6-b66f-41c3-a4e5-c23932004574), while finally converges to a normal distribution:
![Lora B lastlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/dc000056-50c9-46f2-a2ed-f16c7c7e4106). 

From this observation, it implies us that zero initialization is not a good initialization strategy for Lora B in AdamW, instead a normal initialization seems a good strategy since Lora B finally converges to normal distribution. However, as we know, Lora to ensure the constancy of the initialization, one of A,B to choose the all-zero initialization. In our experiment, our results prove that A or B is best with a normal distribution under the AdamW optimizer for both. A remaining question is what kind of normal distribution initialization is appropriate?

## CAME
In CAME, both Lora A and Lora B becomes obvious normal distribution after one training update, and converges to the narrow normal distribution, while B has a smaller standard deviation.
![Lora A lastlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/e3f36aed-a234-4f81-9ec7-742d340c1f8f)
![Lora B lastlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/b3d05911-de6a-4cba-a16a-92492ed7d1a1).
This may imply that we should use normal initialization to replace zero initialization for Lora B.

## adafactor

As for adafactor, Lora A changes unobviously to a normal distribution, while Lora B changes slowly to a concave distribution. In addition, the second last layer converges better than the last layer in the adafactor, and the last layer is nearly converging. Thus, we think more training rounds, and assign Lora A with normal initialization while Lora B with uniform initialization may lead to a better performance. As for learning rate, we alreadly apply the specific AdafactorSchedule.
![Lora A secondlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/253498df-c1ef-41a8-8fc2-23ca9d37aaff)
![Lora B secondlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/3b271834-16e8-47bb-a887-269d84093871)

----
Similar observation can be found in fnn layer. It seems that fnn converges slower than attn layer. For example, the Lora B of adafactor operator in last layer, attn alreadly changes from normal to concave distribution.
![Lora B lastlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/ab258cab-0cb1-49a8-a8db-ffcf7c878234)
![Lora B lastlayer - Epoch 42](https://github.com/L3030/gpt2-lora-practice/assets/74963049/8c4a11cd-3408-4ff0-82df-a52d198026a9)
