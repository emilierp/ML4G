# TO DO 

w1d4 search engine
neel induction heads


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

The Illustrated Transformer
https://jalammar.github.io/illustrated-transformer/

The Illustrated GPT-2 (Visualizing Transformer Language Models)
https://jalammar.github.io/illustrated-gpt2/

How GPT3 Works - Visualizations and Animations
https://jalammar.github.io/how-gpt3-works-visualizations-animations/

Gears-Level Mental Models of Transformer Interpretability
https://www.greaterwrong.com/posts/X26ksz4p3wSyycKNB/gears-level-mental-models-of-transformer-interpretability

A High Level Introduction to Transformers
https://docs.google.com/document/d/1XJQT8PJYzvL0CLacctWcT0T5NfL7dwlCiIqRtdTcIqA/edit#heading=h.icw2n6iwy2of


- Transformer implementation 
- Dataset, DataLoader, Training loop, Optimizer
- Reversed sequence prediction

## Day 6 : Reinforcement Learning

Part 1: Key Concepts in RL
https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

Part 2: Kinds of RL Algorithms
https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html

Part 3: Intro to Policy Optimization
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

Playing Atari with Deep Reinforcement Learning
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

- RL : Vanilla Policy Gradient
- DQN
- Torch : log_prob, sample

## Day 7 : Vision Interpretability (Grad CAM)

Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
https://arxiv.org/pdf/1610.02391.pdf

Feature visualization
https://distill.pub/2017/feature-visualization/

PyTorch Hooks 
https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks 


- Grad CAM 
- Feature visualization on ResNet : 
    - Parametrization : Naive, Fourier, whitening
    - Inner neuron optimization : Instead of maximizing the activation of one of the output classes of the neural network, we can maximize the output of an inner neuron.
- Hooks

## Day 8 : GPT Interpretability

Interpreting GPT: the logit lens
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens 

Probing 
https://www.youtube.com/watch?v=HJn-OTNLnoE

Activation Atlas
https://openai.com/research/introducing-activation-atlases

Explainable AI Cheat Sheet
https://ex.pegg.io/

Chris Olah's views on AGI Safety
https://www.alignmentforum.org/posts/X2i9dQQK3gETCyqh2/chris-olah-s-views-on-agi-safety

- Logit Lens : hook on GPT translation
- Probes
- Activation Atlas : visualize the space the neurons jointly represent.