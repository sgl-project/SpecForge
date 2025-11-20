# 🦅 EAGLE3

## 📍 Overview

In the previous speculative decoding practices, we usually choose a small language model from the same family as the draft model. For example, we can use `Llama-3.1-8B-Instruct` as the draft model and `Llama-3.1-70B-Instruct` as the target model. However, this approach is not always feasible because the small language model may not always be available. Thus, researchers have proposed to train a separate small model as the speculator, this type of models usually use the target model's hidden states or KV cache as input to predict the next few tokens.

Among this type of models, EAGLE3 is the state-of-the-art and has been integrated in [SGLang](https://github.com/sgl-project/sglang). It relies on the hidden states of the target model and often consists of only one dense decoder layer. Before you read on, you can revisit the details of [speculative decoding](./speculative_decoding.md) first if not familiar.

## 🔧 How it works?

<p align="center">
  <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-eagle-drafting-mechanism.gif" alt="EAGLE3"><br>
  <span>Source: <a href="https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/">Blog by NVIDIA</a></span>
</p>

The workflow of EAGLE3 is shown in the animation above. It differs from other speculative decoding methods in several ways:
1. **`Feature-based Drafting`**: Unlike other speculative decoding methods which directly feeds the tokens to the draft model to generate predictions, EAGLE3 operates in the feature space. It will extract the 3 hidden states from the target model at 3 layers at different depths and concatenate them together to form a single feature vector. This feature vector will be fed to the draft model to generate predictions.
2. **`Training-time Test`**: During training, EAGLE3 simulate the autoregressive generation process by autoregressively generating the next few tokens. It then  computes the loss between the predicted output sequence and the ground truth sequence. This method improves the draft model performance because it reduces the generation errors accumulated from previous tokens for higher acceptance rate.
3. **`Dynamic Draft Tree`**: EAGLE3 uses a dynamic draft tree to store the candidate tokens as proposed in [EAGLE2](https://arxiv.org/abs/2406.16858). In simple words, it will only store the candidate tokens that are most likely to be accepted by the target model to improve the acceptance rate.
