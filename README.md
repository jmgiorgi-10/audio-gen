# audio-gen

**Approach**

Current high-level idea is to train a variational autoencoder model from an opensource dataset (specifcally, the Mozilla Common Voice Corpus), and then use the devices microphone to fine-tune the model in real time (with transfer learning),
and adapt to your individual voice. 

**Prerequisites**

Activate a virtual environment for <=Python3.11 and install required dependencies.

```bash
python3 -m venv mlenv
source mlenv/bin/activate
pip3 install numpy
pip3 install torch
```

Variational autoencoder (VAE) consists of two primary components:
1. **Encoder**: Maps the input data $\mathbf{x}$ to a probabilistic distribution over the latent variables \( \mathbf{z} \), typically a Gaussian distribution \( \mathcal{N}(\mu, \sigma^2) \).
2. **Decoder**: Reconstructs the input \( \mathbf{x} \) from samples of the latent variables \( \mathbf{z} \).

### The Variational Approach


### Core Idea
A VAE consists of two primary components:
1. **Encoder**: Maps the input data \( \mathbf{x} \) to a probabilistic distribution over the latent variables \( \mathbf{z} \), typically a Gaussian distribution \( \mathcal{N}(\mu, \sigma^2) \).
2. **Decoder**: Reconstructs the input \( \mathbf{x} \) from samples of the latent variables \( \mathbf{z} \).

### The Variational Approach

The key idea behind VAE is to approximate the true posterior distribution $ p(\mathbf{z} | \mathbf{x}) $ of the latent variables given the data, which is typically intractable. Instead, VAE introduces an **approximating distribution** \( q_{\phi}(\mathbf{z} | \mathbf{x}) \), where \( \phi \) are the parameters of the encoder.

The goal is to maximize the **Evidence Lower Bound (ELBO)**, which is a lower bound on the log-likelihood of the observed data. The ELBO is given by:

\[
\log p(\mathbf{x}) \geq \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right] - \text{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p(\mathbf{z}))
\]

Where:
- \( p_{\theta}(\mathbf{x} | \mathbf{z}) \) is the likelihood of the data given the latent variables (the decoder).
- \( q_{\phi}(\mathbf{z} | \mathbf{x}) \) is the approximating posterior (the encoder).
- \( p(\mathbf{z}) \) is the prior distribution on the latent variables (usually a standard Gaussian \( \mathcal{N}(0, I) \)).
- \( \text{KL} \) denotes the **Kullback-Leibler divergence**, which measures how much \( q_{\phi}(\mathbf{z} | \mathbf{x}) \) deviates from the prior \( p(\mathbf{z}) \).

### Loss Function

The VAE loss function combines two terms:
1. **Reconstruction Loss**: Measures how well the decoder reconstructs the input data \( \mathbf{x} \) from the latent variables \( \mathbf{z} \). It is typically the negative log-likelihood of the data under the decoder, often using mean squared error or binary cross-entropy.
   
2. **KL Divergence Loss**: Regularizes the encoder by encouraging the posterior distribution \( q_{\phi}(\mathbf{z} | \mathbf{x}) \) to be close to the prior distribution \( p(\mathbf{z}) \).

The final VAE loss is:

\[
\mathcal{L} = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right] - \text{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p(\mathbf{z}))
\]
