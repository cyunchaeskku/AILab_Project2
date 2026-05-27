# Project 2: Image Generation — Detailed Course Project Overview

This document provides a comprehensive breakdown of the requirements, constraints, and strategies for **Project 2 (Image Generation)** in the **2026-1 Open-Source AI Practice** course at Sungkyunkwan University (SKKU), taught by Professor Kyuhong Shim. All details are compiled directly from the official slides provided in `project02.pdf`[cite: 1].

---

## 1. Project Task & Objectives

The goal of this project is to build and train a Generative Adversarial Network (GAN) to generate high-quality, realistic human faces.

* **Target Output:** High-resolution $1024 \times 1024$ RGB images.
* **Core Architecture:** The network operates on the standard adversarial paradigm:
  * **Generator ($G$):** Takes a 512-dimensional random noise vector $z \sim N(0, I)$ as input and maps it to a $3 \times 1024 \times 1024$ face image.
  * **Discriminator ($D$):** Evaluates real images from the dataset ($x$) and fake images from the generator ($\hat{x} = G(z)$), outputting scalar validity scores.
* **Adversarial Objectives:**
  1. $D$ must learn to maximize the scores assigned to real images and minimize scores assigned to fake (generated) images.
  2. $G$ must learn to generate realistic images to maximize the $D$ scores of its generated outputs.

---

## 2. Dataset Specifications (FFHQ-1024)

* **Dataset:** Flickr-Faces-HQ (FFHQ) curated human face dataset.
* **Data Split:**
  * **Training Set:** 50,000 images.
  * **Validation Set:** 10,000 images.
  * **Test Set:** 10,000 images.
* **Data Restriction:** **DO NOT USE VALID or TEST for training**.
* **Data Format:** Due to the 89GB size of the original dataset, a JPEG-compressed version is provided to save disk space. While this introduces JPG artifacts, it is acceptable. Both training and evaluation will share the exact same JPG compressed images.
* **External Data:** Do not use external datasets.

---

## 3. Project Constraints & Rules

### Model Parameter Constraints
* **Generator Size:** The total parameter count of the Generator ($G$) **must be under 40M** (Hard threshold).
* **Discriminator Size:** There are no strict parameter count constraints for the Discriminator ($D$), but it must be trained together with $G$.

### Allowed Frameworks & Libraries
* **AI/Deep Learning:** PyTorch and TorchVision.
* **Vision Processing:** OpenCV, Pillow, Scikit-Image, Matplotlib.
* **Experiment Tracking:** Weights & Biases (WandB).
* **Evaluation:** `pytorch-fid`.

### Prohibited Actions & Libraries
* **Do Not Use:** HuggingFace libraries (`Datasets`, `TIMM`, `Hub`, `Evaluate`, etc.).
* **Do Not Use:** Third-party wrapper libraries supporting modeling, automated data pipelines, augmentations, training loops, or evaluations (e.g., PyTorch Lightning, Accelerate, Albumentations).
* **Code Provenance:** **DO NOT START FROM EXISTING REPOSITORIES** and **DO NOT USE PRETRAINED MODELS**.

### AI Usage Policy
* You are highly encouraged to get help from generative AI tools such as ChatGPT, Claude Code, or Gemini. 
* You must briefly document which AI tools you used and how they helped you in your final report.

---

## 4. Suggested Technical Strategies

Training a high-resolution GAN from scratch can be unstable and highly challenging. `project02.pdf` highlights the following high-resolution training frameworks to guide implementation:

### Progressive Growing (PG-GAN Strategy)
* Start training at a heavily downsampled scale ($4 \times 4$ or $8 \times 8$) to learn stable base distributions.
* Progressively insert double-resolution layers into both networks as training proceeds ($256 \rightarrow 512 \rightarrow 1024$).

### Multi-Stage/Refiner Strategy
* Generate mid-resolution features ($256 \times 256$ or $512 \times 512$) using a base generator.
* Use structural Upsampling modules or independent Super-Resolution (SR) / Refiner sub-networks to scale features up to the target $1024 \times 1024$ output.
* *Note:* A relatively weak, pre-baked 256 generator will be provided by the instructor later to support these multi-stage configurations.

### Architectural References
Reviewing architectures such as **BigGAN**, **SA-GAN**, **StyleGAN v2**, and **StyleGAN v3** is highly recommended.

---

## 5. Evaluation & Grading Criteria

The project is worth **35 points total**, distributed across quantitative performance metrics and qualitative evaluation:

### Quantitative Scores (15 Points)
* **FID Score (10 points):** Measures statistical distance to check image quality/diversity.
* **Parameter Count (5 points):** Enforces the hard parameter limit ($G < \text{40M}$).

### Qualitative Scores (10 Points)
* **Professor's Quality Assessment (5 points):** Manual inspection of generation quality and diversity.
* **Students' Quality Assessment (5 points):** Peer evaluation of your generated samples.

### Operational and Implementation Quality (10 Points)
* **Code Quality (5 points):** Requires a clear modular structure, reproducible training and evaluation scripts, reliable checkpoint saving/loading mechanics, readable code, and no hard-coded test-set assumptions.
* **Report Quality (5 points):** Evaluated on the depth of the model architecture explanation, training recipe documentation, validation trajectories, ablation studies/trial history, failure analysis (e.g., artifacts, weak classes), and clear WandB log captures.

---

## 6. Leaderboard & Evaluation Server

* **Timeline:** The automated evaluation leaderboard runs from **May 20 to June 9**. Peer and instructor qualitative reviews take place between **June 10 and June 17**.
* **Deployment Format:** Submissions require exporting the Generator to an **ONNX model** and uploading it to the evaluation server.
* **Server Infrastructure:** The evaluation server automatically generates sample human faces on-the-fly, computes parameter counts, calculates real-time FID metrics, and displays output samples.
* **Overwriting Policy:** The leaderboard accepts exactly *one active model entry* per student. You may iterate and re-upload models freely. However, updating your model will reset previous qualitative evaluation scores, so your architecture must be finalized when peer-grading begins.

---

## 7. Final Submission Instructions

### Schedule Deadlines
* **Project Due Date:** **June 9, 2026, at 23:59 KST**.

### Archive Structure
Compress your full codebase, your optimized generator checkpoint file, and your technical PDF report into a single consolidated file named `2025xxxxxx_project02.zip`. **Do not submit the discriminator**.


---

## 8. Appendix: Architectural Reference Summaries

### BigGAN: Large Scale GAN Training for High Fidelity Natural Image Synthesis (ICLR 2019)
Focuses on the benefits of scaling GAN training.
*   **Scaling:** Increases batch sizes (up to 2048) and channel counts significantly. Large batches provide better gradients by covering more modes.
*   **Architectural Changes:** Introduces shared class embeddings (projected to each layer) and "Skip-z" connections (latent chunks concatenated to intermediate layers) to improve parameters and training speed.
*   **Truncation Trick:** Samples latent vectors from a truncated normal distribution. Lower truncation thresholds improve individual sample fidelity at the cost of overall variety.
*   **Regularization:** Uses Orthogonal Regularization to make the generator more amenable to truncation by enforcing smoothness.

### SA-GAN: Self-Attention Generative Adversarial Networks (ICML 2019)
Introduces attention mechanisms to capture global dependencies.
*   **Self-Attention:** Overcomes the limitations of local receptive fields in convolutions. The attention module calculates the response at a position as a weighted sum of features at all positions.
*   **Stabilization:** Applies Spectral Normalization (SN) to both the Generator and Discriminator. SN restricts the spectral norm of weights to 1, preventing parameter explosion and unusual gradients.
*   **TTUR:** Recommends the Two-Timescale Update Rule (different learning rates for G and D) to compensate for slow learning in regularized discriminators.

### StyleGAN v2: Analyzing and Improving the Image Quality of StyleGAN (CVPR 2020)
Focuses on removing characteristic artifacts and improving training stability.
*   **Weight Demodulation:** Replaces Adaptive Instance Normalization (AdaIN) to eliminate "droplet artifacts." Normalization is based on expected statistics of weights rather than actual feature map contents.
*   **No Progressive Growing:** Replaces the unstable progressive growing schedule with skip connections in the Generator and residual connections in the Discriminator.
*   **Path Length Regularization:** Encourages fixed-magnitude changes in the output for fixed-size steps in latent space, resulting in smoother latent space mappings and easier model inversion.
*   **Capacity:** Identified resolution under-utilization, leading to increased model capacity for better high-frequency detail.

### StyleGAN v3: Alias-Free Generative Adversarial Networks (NeurIPS 2021)
Redesigns the synthesis process to be fully equivariant to translation and rotation.
*   **Alias-Free:** Identifies aliasing (from upsampling and nonlinearities) as the cause of "texture sticking." Interprets signals as continuous and band-limited.
*   **Filters:** Replaces standard up/downsampling with high-quality windowed Kaiser filters (>100dB attenuation) to strictly enforce band-limiting.
*   **Filtered Nonlinearities:** Wraps nonlinearities (like Leaky ReLU) between upsampling and downsampling to suppress high frequencies generated by the pointwise operation.
*   **StyleGAN3-R:** A rotation-equivariant version using 1x1 convolutions and radially symmetric filters (jinc-based).