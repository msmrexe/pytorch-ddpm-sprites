# Scratch-Built Diffusion Models: DDPM & DDIM for Sprites

Diffusion models have rapidly become a cornerstone of modern generative AI, known for their ability to produce stunningly high-fidelity results. This project provides a complete **from-scratch PyTorch implementation** exploring the core mechanics of these powerful models. It implements the foundational **Denoising Diffusion Probabilistic Model (DDPM)** and its faster, deterministic counterpart, the **Denoising Diffusion Implicit Model (DDIM)**. Developed for the M.S. course Generative Models, this repository breaks down the complex theory into clean, modular code for generating 16x16 pixel art sprites.

## Features

* **Denoising Diffusion Probabilistic Model (DDPM)**: Full implementation from scratch.
* **Denoising Diffusion Implicit Models (DDIM)**: Includes a faster, deterministic DDIM sampling loop.
* **U-Net Noise Predictor**: A U-Net architecture designed to predict the noise added at any timestep.
* **Modular Code**: All logic is organized into a clean, importable `src/` package.
* **Evaluation**: Built-in script to calculate the Fréchet Inception Distance (FID) score.

## Core Concepts & Techniques

* **Generative Modeling**: Learning a data distribution $p(x)$ to generate new samples.
* **Diffusion Models**: A class of models that work by systematically destroying data structure (forward process) and then learning to reverse the process (reverse process).
* **Forward (Noising) Process**: A Markov process that gradually adds Gaussian noise to an image $\mathbf{x}_0$ over $T$ timesteps, producing a sequence of noisy images $\mathbf{x}_1, ..., \mathbf{x}_T$.
* **Reverse (Denoising) Process**: A learned Markov process $p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)$ that denoises an image from $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ back to a clean image $\mathbf{x}_0$.
* **U-Net Architecture**: Using skip connections to preserve high-resolution features, making it ideal for image-to-image tasks like noise prediction.

---

## How It Works

This project trains a model, $\epsilon_{\theta}$, to reverse a diffusion process. The process is broken into two parts: the fixed forward process and the learned reverse process.

### 1. The Forward (Noising) Process

The forward process, $q$, gradually adds Gaussian noise to a clean image $\mathbf{x}\_0$ according to a variance schedule $\beta_t$. We define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod_{i=1}^{t} \alpha_i$.

A key property of this process is that we can sample $\mathbf{x}_t$ at any arbitrary timestep $t$ in a closed-form equation, without having to iterate through all $t$ steps:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

This means we can generate a training pair $(\mathbf{x}_t, t)$ by picking a random image $\mathbf{x}_0$, a random timestep $t$, and sampling a noise vector $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. The noised image is then:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

### 2. The Learned Reverse (Denoising) Process

The goal of the model is to learn the reverse process $p_{\theta}(\mathbf{x}\_{t-1} | \mathbf{x}\_t)$. It can be shown that if $\beta_t$ is small, this reverse transition is also Gaussian. The model $\epsilon_{\theta}(\mathbf{x}\_t, t)$ is trained to predict the noise $\epsilon$ that was added to create $\mathbf{x}_t$.

The training loss is a simple Mean Squared Error (MSE) between the predicted noise and the actual noise:

$$L = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ ||\epsilon - \epsilon_{\theta}(\mathbf{x}_t, t)||^2 \right]$$

### 3. Sampling (DDPM vs. DDIM)

Once the model $\epsilon_{\theta}$ is trained, we can generate new images by starting with pure noise $\mathbf{x}\_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively sampling $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$ for $t = T, ..., 1$.

#### Algorithm 1: DDPM Sampling

The original DDPM paper derives the following equation for sampling $\mathbf{x}_{t-1}$:

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$

where $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ (if $t > 1$) and $\sigma_t^2 = \beta_t$. This is a **stochastic** process, as new noise $\mathbf{z}$ is added at each step. It requires all $T$ steps (e.g., 1000) to generate an image.

#### Algorithm 2: DDIM Sampling

DDIM provides a more general sampling process that is **deterministic** when $\eta = 0$. It also allows for "jumps," sampling in far fewer steps (e.g., 50-100) while achieving high-quality results.

The DDIM update rule is:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0^{\text{pred}} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_{\theta}(\mathbf{x}_t, t) + \sigma_t \mathbf{z}$$

where $\mathbf{x}_0^{\text{pred}}$ is the model's prediction of the *original* clean image, and $\sigma_t$ is a parameter controlled by $\eta$. When $\eta=0$, $\sigma_t=0$, and the process becomes deterministic.

### 4. Model Architecture (U-Net)

The noise predictor $\epsilon_{\theta}(\mathbf{x}_t, t)$ is a U-Net.
* **Input:** A noised image $\mathbf{x}_t$ (shape `[B, 3, 16, 16]`) and its timestep $t$.
* **Output:** The predicted noise $\epsilon$ (shape `[B, 3, 16, 16]`).
* **Architecture:** It consists of a down-sampling path (encoder) and an up-sampling path (decoder) with skip connections. The timestep $t$ and context labels $c$ are embedded and injected into the model at various resolutions. This implementation uses `ResidualConvBlock`s and fixes a critical inefficiency from the original notebook where a shortcut layer was re-initialized on every forward pass.

---

## Project Structure

```
pytorch-diffusion-sprites/
├── .gitignore             # Ignores data, logs, outputs, and pycache
├── LICENSE                # MIT License file
├── README.md              # You are here!
├── requirements.txt       # Project dependencies
├── notebooks/
│   └── run.ipynb          # Jupyter notebook to run the full pipeline
├── scripts/
│   ├── download_data.sh   # Script to download the .npy dataset
│   ├── train.py           # Main training script
│   ├── sample.py          # Script to generate sample images
│   └── evaluate.py        # Script to generate images and run FID evaluation
└── src/
    ├── __init__.py        # Makes 'src' a Python package
    ├── config.py          # All hyperparameters and file paths
    ├── data_loader.py     # CustomDataset and get_dataloaders function
    ├── model.py           # U-Net model architecture (Unet, ResidualConvBlock, etc.)
    ├── diffusion.py       # DiffusionScheduler class (holds DDPM/DDIM logic)
    └── utils.py           # Utility functions (logging, plotting, saving images)
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-diffusion-sprites.git
    cd pytorch-diffusion-sprites
    ```

2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Data:**
    Run the download script. This will create a `data/` folder and place the `.npy` files inside.
    ```bash
    bash scripts/download_data.sh
    ```

4.  **Train the Model:**
    Run the training script. The model will be trained according to the settings in `src/config.py`. The best model (based on validation loss) will be saved to `outputs/models/ddpm_sprite_best.pth`. A loss plot will be saved to `outputs/loss_plot.png`.
    ```bash
    python scripts/train.py
    ```

5.  **Generate Samples:**
    After training, you can generate a grid of sample images.

    * **Using DDPM (1000 steps, stochastic):**
        ```bash
        python scripts/sample.py --n-samples 16 --method ddpm
        ```
    * **Using DDIM (50 steps, deterministic):**
        ```bash
        python scripts/sample.py --n-samples 16 --method ddim --n-ddim-steps 50 --eta 0.0
        ```
    This will save a file to `outputs/samples/`.

6.  **Evaluate the Model (FID Score):**
    This script will generate 3000 real images and 3000 fake images, save them to `outputs/eval/`, and then compute the FID score.
    ```bash
    python scripts/evaluate.py --n-samples 3000 --method ddim --n-ddim-steps 100
    ```
<!---
    *Expected Output:*
    ```
    [...]
    [2025-10-31 10:11:18] [INFO] Calculating FID score...
    [2025-10-31 10:12:18] [INFO] *** FID Score: [some_value] ***
    [2025-10-31 10:12:18] [INFO] Evaluation complete.
    ```
--->

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
