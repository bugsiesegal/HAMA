# HAMA: Hierarchical Attention-based Masked Autoencoder

> "Hama early, Hama often." - Joseph Anderson, 2021

HAMA is an advanced transformer-based architecture that combines hierarchical processing, parallel masked autoencoders, and attention mechanisms to learn robust representations of sequential data.

## Features

- **Hierarchical Processing**: Multi-layer architecture processing data at different scales
- **Parallel Processing**: Multiple specialized nodes per layer for efficient computation
- **Adaptive Masking**: Learned masking strategies for robust feature extraction
- **Fission-Fusion Mechanism**: Smart splitting and recombination of information streams
- **Flexible Scaling**: Configurable architecture through scaling factors or explicit parameters

## Installation

```bash
pip install torch  # Required dependency
git clone https://github.com/bugsiesegal/hama.git
cd hama
pip install -e .
```

## Quick Start

```python
import torch
from hama import HAMA

# Initialize HAMA model
model = HAMA(
    embedding_dim=256,
    num_heads=8,
    dropout=0.1,
    num_layers=3,
    transformer_layers=2,
    initial_num_nodes=8,
    initial_partition_length=64,
    initial_n_mask=4,
    num_nodes_scaling_factor=0.5,
    partition_length_scaling_factor=2.0
)

# Example input
x = torch.randn(32, 512, 256)  # [batch_size, seq_len, embedding_dim]

# Forward pass
output = model(x)

# Encode only
encoded = model.encode(x)

# Decode only
decoded = model.decode(encoded)
```

## Architecture Overview

HAMA consists of several key components working together:

1. **Hierarchical Layers**
   - Multiple processing layers operating at different scales
   - Progressive information abstraction
   - Bidirectional information flow

2. **Processing Nodes**
   - Parallel autoencoders within each layer
   - Specialization in different aspects of input
   - Decreasing count in higher layers

3. **Masking Mechanism**
   - Token importance-based masking
   - Progressive masking through layers
   - Forced robust feature learning

4. **Fission-Fusion Process**
   - Input splitting via learned queries
   - Parallel stream processing
   - Attention-based recombination

## Configuration

HAMA offers two ways to configure its architecture:

### 1. Scaling Factors

```python
model = HAMA(
    embedding_dim=256,
    num_heads=8,
    dropout=0.1,
    num_layers=3,
    transformer_layers=2,
    initial_num_nodes=8,
    initial_partition_length=64,
    initial_n_mask=4,
    num_nodes_scaling_factor=0.5,
    partition_length_scaling_factor=2.0,
    n_mask_scaling_factor=1.0
)
```

### 2. Explicit Layer Configuration

```python
model = HAMA(
    embedding_dim=256,
    num_heads=8,
    dropout=0.1,
    num_layers=3,
    transformer_layers=2,
    nodes_per_layer=[8, 4, 2],
    partition_lengths=[64, 128, 256],
    n_masks=[4, 4, 4]
)
```

## Model Components

### SinusoidalPositionalEncoding
Adds positional information to input embeddings using sine and cosine functions.

### BabyAutoencoder
Simple transformer-based autoencoder with token masking capability.

### FissionModule
Splits input sequence into multiple node-specific representations.

### FusionModule
Combines multiple node representations back into a single sequence.

### HAMABlock
Combines fission, node-specific processing, and fusion operations.

## Training Tips

1. **Start Simple**
   - Begin with fewer layers and nodes
   - Gradually increase complexity

2. **Monitor Masking**
   - Watch reconstruction quality
   - Adjust masking ratios if needed

3. **Scaling Considerations**
   - Larger partition lengths in higher layers
   - Fewer nodes in higher layers
   - Balance between compression and reconstruction

4. **Attention Heads**
   - Use multiple heads for better feature capture
   - Typically 8-16 heads work well

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use HAMA in your research, please cite:

```bibtex
@article{segal2025hama,
  title={HAMA: Hierarchical Attention-based Masked Autoencoder},
  author={Jake Segal},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions and feedback, please open an issue on GitHub or contact bugsie@segalnyc.com.
