# EchoMemoryAttention vs. Multi-Head Attention (MHA) Benchmarking

This experiment evaluates a novel attention mechanism called **EchoMemoryAttention** against standard PyTorch **Multi-Head Attention (MHA)** on synthetic long-context sequence processing and language modeling tasks.

## 1. Architectural Overview

### Standard Multi-Head Attention (MHA)
*   Projects inputs into Query, Key, and Value spaces.
*   Computes dense pairwise dot-product attention scores across all memory positions.
*   Lacks dynamic pruning or gating, incurring high computationally complex footprints over long contexts.

### Dynamic Sigmoid Echo Attention
*   Utilizes **Shannon Entropy** over attention weights to scale the number of top-voted memories ($k$) retrieved dynamically.
*   Higher context uncertainty automatically scales $k$ up for broader memory coverage; confidence scales $k$ down to prune noise.
*   Applies a sigmoid gate over combined inputs for residual vector injection.

### Dynamic SwiGLU Echo Attention
*   Enhances the Echo Attention block by replacing standard gating with a **SwiGLU (Swish Gated Linear Unit)** feed-forward gating formulation: $\text{Swish}(x W) \otimes (x V)$.
*   Maintains the dynamic entropy-driven selection mechanism while improving gradient flow and model capacity.

---

## 2. Experimental Scenarios

1.  **Standard Context & Deeper Architectures ($N=3$ Layers)**: Evaluating baseline convergence speed on clean datasets.
2.  **Noise & Distractor Regimes**: Simulating distraction by introducing up to 80% noise (distractor) tokens into the sequence contexts.
3.  **Generalization Audits**: Benchmarking train vs. validation convergence to study overfitting behaviors.
4.  **Parameter Capacity Scaling**: Stress-testing the model with $120,000$ sequences to evaluate training throughput and prevent memorization.
5.  **Inference Latency & FLOP Audits**: Micro-benchmarking processing speed (ms/token) and parameter counts.

---

## 3. Empirical Results Summary

*   **Loss Convergence**: The **Dynamic SwiGLU Echo** model shows superior convergence rate and final loss performance (Final Cross-Entropy Loss ~ 0.10) under large-scale noise tasks, matching or exceeding Standard MHA while utilizing highly pruned attention representations.
*   **Parameter & FLOP Profile**:
    *   *Standard MHA*: 16,640 parameters, ~32,768 FLOPs/token.
    *   *Dynamic Sigmoid*: 6,241 parameters, ~14,976 FLOPs/token.
    *   *Dynamic SwiGLU*: 20,480 parameters, ~39,360 FLOPs/token.
*   **Inference Latency**: Standard MHA averages 0.31 ms/token, whereas Dynamic SwiGLU Echo runs at 0.40 ms/token (a 1.27x overhead) due to non-vectorized Python top-k masking.
