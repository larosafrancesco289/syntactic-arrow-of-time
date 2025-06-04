# Investigation on the Effect of Grammar on the Arrow of Time Using Large Language Models

This repository contains a PyTorch implementation of a GPT (Generative Pre-trained Transformer) model designed to investigate how grammatical structure affects temporal directionality in language models. The implementation is based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and has been modified to work with Part-of-Speech (POS) tag sequences rather than raw text.

## Overview

This project explores whether large language models exhibit different behaviors when processing sequences in forward vs. backward directions, particularly focusing on how grammatical patterns (represented as POS tags) influence this "arrow of time" phenomenon. The research investigates temporal asymmetries in language model predictions using controlled experiments with grammatical structures.

## Report

📄 **[Research Report: Investigation on the Effect of Grammar on the Arrow of Time Using Large Language Models](Investigation_on_the_Effect_of_Grammar_on_the_Arrow_of_Time_Using_Large_Language_Models.pdf)**

The complete research findings, methodology, and experimental results are documented in the above report.

## Architecture

The model follows the standard GPT architecture with the following key modifications:

- **Token-level Processing**: Instead of subword tokens, the model processes sequences of POS (Part-of-Speech) tags
- **Bidirectional Training**: Support for both forward and backward sequence processing
- **Multiple Language Support**: Datasets for various languages including English, German, Italian, and Finnish

## Project Structure

```
├── train.py              # Main training script with DDP support
├── model.py              # GPT model implementation
├── generate.py           # Text generation from POS sequences
├── dataloader.py         # Custom dataset loader for POS sequences
├── sentence_loss.py      # Sentence-level evaluation utilities
├── predict_loss.py       # Loss prediction and analysis
├── configurator.py       # Configuration management
├── pos_dictionary.json   # POS tag to word mappings
├── requirements.txt      # Python dependencies
├── data/                 # Dataset directory
│   ├── openwebtext/      # OpenWebText dataset
│   ├── cc100_*/          # Common Crawl multilingual datasets
│   └── shakespeare_*/    # Shakespeare corpus variations
└── config/               # Configuration files for different experiments
    ├── bpe/              # BPE tokenization configs
    ├── id_model/         # ID-based model configs
    ├── range_model/      # Range-based model configs
    └── extra_lang/       # Additional language configs
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

### Basic Usage

#### Training a Model

```bash
# Single GPU training
python train.py --batch_size=32 --compile=False

# Multi-GPU training with DDP
torchrun --standalone --nproc_per_node=4 train.py
```

#### Generating Text

```bash
python generate.py --out_dir=out --num_samples=5 --max_new_tokens=100
```

#### Evaluating Sentence-Level Loss

```bash
python sentence_loss.py --checkpoint_gpt1=out/ckpt.pt --meta_path=data/openwebtext/meta.pkl
```

## Datasets

The project supports multiple datasets for different experimental conditions:

- **OpenWebText**: English web text corpus
- **CC-100**: Multilingual Common Crawl data (German, Italian, Finnish, etc.)
- **Shakespeare**: Literary text for controlled experiments
- **Custom POS sequences**: Preprocessed grammatical pattern datasets

Each dataset includes:
- Tokenized sequences (`train.bin`, `val.bin`)
- Metadata with vocabulary mappings (`meta.pkl`)
- Configuration files for reproducible experiments

## Research Applications

This implementation enables investigation of:

- **Temporal Directionality**: How models process time-ordered sequences
- **Grammatical Constraints**: Impact of syntactic structure on predictions
- **Cross-Lingual Patterns**: Universal vs. language-specific temporal biases
- **Model Scale Effects**: How model size affects temporal processing

## Acknowledgments

This implementation is based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), which provides an excellent foundation for GPT experimentation. We extend our gratitude to Karpathy for making high-quality educational implementations accessible to the research community.

The core transformer architecture and training loop are adapted from nanoGPT, with significant modifications for:
- POS tag processing
- Bidirectional sequence handling
- Multilingual dataset support
- Grammar-aware text generation

## Citation

If you use this code in your research, please cite both this work and the original nanoGPT implementation:

```bibtex
@misc{nanogpt,
  author = {Andrej Karpathy},
  title = {nanoGPT},
  year = {2022},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/karpathy/nanoGPT}},
}
```

## License

This project follows the same MIT license as the original nanoGPT implementation.