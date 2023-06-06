# Day 1 

https://einops.rocks/1-einops-basics/

https://einops.rocks/2-einops-for-deep-learning/

- PyTorch Exercise (w0d1 mlab)

- Einops (rearrange, reduce, repeat), Torch (arange, einsum, as_strided), Indexing (integer based, torch.gather)

- ReLU, Batched Log-Softmax, Log-CrossEntropy loss, LogSumExp, Sample distribution

## Day 2 : ResNet

ResNet https://arxiv.org/pdf/1512.03385.pdf

Batch Normalization in Convolutional Neural Networks  https://www.baeldung.com/cs/batch-normalization-cnn

- ResNet : Data preparation (troch.Transforms, Compose), Predictions (model.eval(), with torch.inference_mode(), softmax, topk)

- Practises with einsum and as_strided

- Functions (Convolution, Padding, Stride, MaxPooling), Modules (Convolution, Linear, Pooling, Batch Normalization)


## Day 3 : Adversarial Attacks

- Binary Classification and training loop with hyperparameter tuning

- Adversarial attacks : FSGM, Patch attack

## Day 4 : Optimizers

- Optimizer : Adam, SGD

- Image Memorizer : DataLoader, Dataset, Train-Test Split by indexing, visualization

## Day 5 : Transformer

https://jalammar.github.io/illustrated-transformer/
https://jalammar.github.io/illustrated-gpt2/
https://jalammar.github.io/how-gpt3-works-visualizations-animations/

https://www.greaterwrong.com/posts/X26ksz4p3wSyycKNB/gears-level-mental-models-of-transformer-interpretability
https://docs.google.com/document/d/1XJQT8PJYzvL0CLacctWcT0T5NfL7dwlCiIqRtdTcIqA/edit#heading=h.icw2n6iwy2of


- Transformer implementation 
- Dataset, DataLoader, Training loop, Optimizer
- Reversed sequence prediction

## Day 6 : Reinforcement Learning

https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

- RL : Vanilla Policy Gradient
- DQN
- Torch : log_prob, sample