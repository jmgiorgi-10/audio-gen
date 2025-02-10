# audio-gen

**Aproach**

The idea is to train a variational autoencoder model from an opensource dataset (specifcally, the Mozilla Common Voice Corpus), and then use your devices to fine-tune the model in real time (with transfer learning),
and adapt to your individual voice. 

**Libraries**

```bash
python3 -m venv mlenv
source mlenv/bin/activate
pip3 install numpy
pip3 install torch
