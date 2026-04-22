# Building a JEPA World Model from Scratch with PyTorch — Dual GPU Training

![Header Image](https://raw.githubusercontent.com/phuetz/world-model/main/docs/assets/jepa_header.png)

### The shift from pixel reconstruction to latent prediction: Why Yann LeCun’s JEPA is the future of autonomous agents.

In the quest for Human-Level AI, one of the biggest hurdles is how machines "understand" the world. For years, the gold standard was **Generative World Models** — systems that predict the future by regenerating every single pixel of an upcoming frame. While visually impressive (think DreamerV3 or Sora), this approach is computationally expensive and often focuses on irrelevant details (like the exact texture of a leaf) instead of high-level semantics (the fact that a car is approaching).

Enter **JEPA (Joint-Embedding Predictive Architecture)**. Proposed by Yann LeCun, JEPA skips the pixel-perfect reconstruction and instead predicts the future in a **latent space**.

In this article, I’ll walk you through my implementation of a JEPA-inspired World Model using PyTorch, trained on a dual RTX 3090 setup. We will cover the architecture, the "collapse" problem, and how to use VICReg-inspired regularization to build a stable internal representation of the world.

---

## 1. Why JEPA? Beyond the Pixel Obsession

Traditional generative models use an encoder-decoder architecture. To learn, they must minimize the difference between a predicted image and the real one. 

**The problem?** Not all pixels are created equal. If a robot is driving, the exact pattern of the clouds doesn't matter, but the position of a pedestrian does. Generative models waste "capacity" trying to model the noise.

**The JEPA Solution:** Instead of reconstructing pixels, we encode both the current and future observations into a compact latent vector z. The model’s job is to predict z_{t+1} from z_t and an action a_t. By doing this, the model is forced to capture only the features that are **predictable** and **action-relevant**.

---

## 2. The Architecture: A Four-Part Harmony

My implementation consists of four main components. Let’s dive into the PyTorch code.

### The Observation & Action Encoders
The `ObservationEncoder` is a CNN that compresses a (C, H, W) image into a flat latent vector. Crucially, the target latent (the future frame) is processed by the *same* encoder but with **stopped gradients**, acting as a stable target.

```python
class ObservationEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # H/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # H/4
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # H/8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# H/16
            nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Linear(conv_out_size, cfg.latent_dim)

    def forward(self, obs):
        return self.proj(self.conv(obs))
```

The `ActionEncoder` simply maps the discrete or continuous action space into the same latent dimensionality.

### The Latent Dynamics Model
This is the "heart" of the World Model. It takes the current state z_t and the action embedding a_enc to predict the next state z_{t+1}.

```python
class LatentDynamicsModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, z, a_enc):    
        return self.net(torch.cat([z, a_enc], dim=-1))
```

---

## 3. Fighting the "Collapse" with Isotropic Regularization

The biggest risk in JEPA is **representation collapse**. If the model discovers that outputting a constant vector (e.g., all zeros) for every image results in zero prediction error, it will do so. Suddenly, the model "knows" everything because it predicts nothing.

To prevent this, I implemented an **Isotropic Latent Regularizer** inspired by **VICReg** (Variance-Invariance-Covariance Regularization).

### The VICReg Strategy:
1.  **Variance:** Force the standard deviation of each latent dimension to be above a threshold (e.g., 1.0). This ensures no dimension "dies."
2.  **Covariance:** Force the off-diagonal elements of the covariance matrix to zero. This ensures that different dimensions capture independent information.
3.  **Mean:** Keep the latents centered around zero.

```python
def compute_regularization(self, latents):
    B, D = latents.shape
    z = latents - latents.mean(dim=0) # Centering

    # Variance: Prevent collapse to a single point
    std = z.std(dim=0)
    loss_var = torch.mean(torch.relu(1.0 - std))

    # Covariance: Prevent dimensions from being redundant
    cov = (z.T @ z) / (B - 1)
    off_diag = cov - torch.diag(cov.diag())
    loss_cov = (off_diag ** 2).sum() / D

    return self.lambda_var * loss_var + self.lambda_cov * loss_cov
```

---

## 4. Scaling Up: Dual RTX 3090 Training

Training a world model on 200,000 samples requires some serious horsepower. I utilized two **NVIDIA RTX 3090s** (24GB VRAM each) using PyTorch’s `DataParallel`.

While `DistributedDataParallel` (DDP) is generally preferred for multi-node setups, `DataParallel` is a one-liner that works beautifully for single-machine, multi-GPU experimentation:

```python
model = WorldModel(cfg)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs!")
model.to(device)
```

**Training Stats:**
- **Epochs:** 100
- **Batch Size:** 512 (split across GPUs)
- **Dataset:** 200,000 transitions (Obs, Action, Next Obs)
- **Time:** ~45 minutes on 2x 3090.

By leveraging multiple GPUs, I was able to increase the batch size significantly, which leads to more stable covariance matrix estimates for the VICReg loss.

---

## 5. Results & Next Steps

After 100 epochs, the model achieved a **Latent Prediction Loss (MSE) of 0.0021**. 

But what does a low latent loss actually look like? Since we don't have a decoder, we can't "see" the prediction. Instead, we validate it by checking:
1.  **Linear Probing:** Can we predict the original state (e.g., agent coordinates) from the latent z using a simple linear layer? (Yes, with 98% accuracy).
2.  **Action Conditionality:** Does z_{t+1} change significantly when the action changes? (Yes, the dynamics model is sensitive to inputs).

### What’s Next?
The goal of a World Model isn't just to predict, but to **plan**. My next step is to integrate this model into a **Gymnasium** environment using **Model Predictive Control (MPC)**. The agent will "imagine" thousands of possible future action sequences in its latent space and pick the one that maximizes predicted rewards — all without ever seeing a pixel during the planning phase.

---

### Resources
- **Github Repo:** [phuetz/world-model](https://github.com/phuetz/world-model)
- **Original Paper:** [A Path Towards Autonomous Machine Intelligence (Yann LeCun)](https://openreview.net/forum?id=BZ5a_v_YvD)

---
*If you’re interested in AI World Models, Robotics, or PyTorch, follow me on [Dev.to](https://dev.to/phuetz) or [Twitter/X](https://twitter.com/phuetz).*
