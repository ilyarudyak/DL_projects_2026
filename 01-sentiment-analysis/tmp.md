While all three are variants of the **Adam (Adaptive Moment Estimation)** family, they handle **weight decay** and **momentum** in fundamentally different ways.

Here is the quick summary:

* **`Adam`**: The standard adaptive optimizer. It combines momentum with adaptive learning rates, but its weight decay implementation is mathematically flawed (it treats weight decay like $L_2$ regularization).
* **`AdamW`**: The corrected version of Adam. It decouples weight decay from the adaptive gradient updates, making weight decay work properly. **This is usually the default choice for deep learning (especially Transformers/LLMs).**
* **`NAdam`**: Adam with **Nesterov momentum**. Instead of looking at past gradients to update weights, it looks "ahead" in the direction of momentum to compute gradients, leading to faster convergence on certain architectures.

---

## 1. `Adam` vs. `AdamW` (The Weight Decay Fix)

In classical SGD, adding $L_2$ regularization $\frac{1}{2}\lambda \vert{}\vert{}\theta\vert{}\vert{}^2$ to the loss function is mathematically equivalent to applying direct weight decay $\theta_t \leftarrow (1 - \eta \lambda)\theta_{t-1}$.

In standard **`Adam`**, $L_2$ regularization was added directly to the gradient $g_t$:

$$g_t \leftarrow g_t + \lambda \theta_{t-1}$$

Because Adam divides updates by $\sqrt{v_t}$ (the second moment / variance estimate), weights with **large gradients get scaled down**, which unintentionally **scales down their weight decay**. Conversely, weights with small gradients end up getting decayed *more*.

**`AdamW`** fixes this by decoupling weight decay from the gradient moments entirely, applying it directly at the final step:

$$\theta_t \leftarrow \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right) - \eta \lambda \theta_{t-1}$$

> **Key Takeaway:** If you set `weight_decay > 0`, almost always use **`AdamW`** over `Adam`. Standard `Adam` fails to regularize parameters with large gradients properly.

---

## 2. `Adam` vs. `NAdam` (Nesterov Accelerated Momentum)

Standard **`Adam`** uses Classical Momentum ($m_t$), which accumulates past gradients to push the step vector in the moving-average direction:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**`NAdam`** incorporates **Nesterov Accelerated Gradient (NAG)**. Instead of taking a step based purely on past momentum, it effectively evaluates the gradient at a look-ahead position $\theta_{t-1} + \beta_1 m_{t-1}$.

In PyTorch's implementation, this is reformulated so you don't need extra forward/backward passes. It blends the current step's bias-corrected gradient vector directly into the momentum prediction vector.

* **Advantage:** Helps prevent overshooting steep minima and speeds up convergence in high-curvature loss landscapes.
* **Trade-off:** Can occasionally be slightly less stable with high learning rates or noisy batch statistics.

---

## Direct Comparison

| Optimizer | Momentum Type | Weight Decay Implementation | Best Used For |
| --- | --- | --- | --- |
| **`Adam`** | Classical ($m_t$) | Coupled ($L_2$ on gradients) | Legacy baselines, `weight_decay=0` cases |
| **`AdamW`** | Classical ($m_t$) | Decoupled (Direct on weights) | **Transformers, LLMs, Vision Transformers, Deep Networks** |
| **`NAdam`** | Nesterov (Look-ahead) | Coupled ($L_2$ on gradients)* | CNNs, RNNs, tasks needing faster early convergence |

**Note: If you want Nesterov momentum with decoupled weight decay, PyTorch provides `torch.optim.NAdam` with a `decoupled_weight_decay` parameter (or you can look at custom/experimental implementations like `AdamW` with Nesterov).*

---

## When to use which?

1. **Default to `AdamW`:** For almost all modern deep learning models (especially Transformers, CNNs, and language models), `AdamW` with `weight_decay=0.01` or `0.1` is the gold standard.
2. **Use `NAdam`:** If you are training deep neural networks without heavy regularization requirements and notice `Adam` converging too slowly near plateau areas.
3. **Use `Adam`:** Only if you are maintaining legacy codebases, running experiments with zero weight decay (where `Adam` and `AdamW` behave identically), or matching older paper implementations.