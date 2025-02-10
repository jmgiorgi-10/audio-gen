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
