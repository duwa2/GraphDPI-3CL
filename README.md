# GraphDPI-3CL
<p align="center">
<img width="612" alt="image" src="https://github.com/duwa2/GraphDPI-3CL/assets/158106190/8193a19c-0789-436d-9268-d0d447919ef2">
</p>

# Molecular Feature Extraction Model with MPNN, BiLSTM, and Self-Attention

This repository contains a PyTorch implementation of a neural network model designed for molecular feature extraction. The model architecture integrates a Message Passing Neural Network (MPNN) for initial feature extraction from molecular graphs, followed by a Bidirectional LSTM (BiLSTM) for sequential data processing, and a Self-Attention mechanism for capturing long-range dependencies. The final output is passed through a Fully Connected (FC) layer.

## Model Overview

The model consists of the following components:

1. **MPNN**: A custom implementation of a Message Passing Neural Network to process graph-structured data.
2. **BiLSTM**: A Bidirectional LSTM layer to handle sequential information.
3. **Self-Attention**: An attention mechanism that allows the model to focus on different parts of the sequence.
4. **Fully Connected Layer**: A dense layer that outputs the final predictions.

## Requirements

To run this model, you will need the following libraries:

- PyTorch3.6
- PyTorch Geometric (torch-geometric)

You can install the required libraries using pip:

```bash
pip install torch torch-geometric
```
```bash
git clone https://github.com/your-username/MPNN-BiLSTM-SelfAttention-Model.git
cd MPNN-BiLSTM-SelfAttention-Model
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Contact
If you have any questions or comments about this repository, please open an issue or contact the repository owner. Weian Du (duwan@mail.sysu.edu.cn)
