# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Toto (Time Series Optimized Transformer for Observability) is a foundation model for multivariate time series forecasting. It uses a decoder-only transformer architecture with alternating time-wise and space-wise attention, designed for zero-shot forecasting on high-dimensional observability data.

## Development Commands

### Testing
```bash
pytest toto/test/                    # Run all tests
pytest toto/test/model/             # Test specific module
pytest -m cuda                     # GPU-specific tests
```

### Code Quality
```bash
black --line-length 120 .          # Format code
isort .                            # Sort imports  
mypy toto/                         # Type checking
```

### Evaluation
```bash
# LSF benchmark evaluation
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$(pwd):$(pwd)/toto:$PYTHONPATH"
python toto/evaluation/run_lsf_eval.py --datasets ETTh1 ETTh2 --context-length 2048 --checkpoint-path Datadog/Toto-Open-Base-1.0
```

## Architecture

### Core Components
- **Model** (`toto/model/`): TotoBackbone with transformer layers, attention mechanisms, embeddings, and distribution heads
- **Inference** (`toto/inference/`): TotoForecaster with autoregressive decoding and KV caching
- **Data** (`toto/data/`): MaskedTimeseries data structure and dataset utilities
- **Evaluation** (`toto/evaluation/`): LSF and GIFT-Eval benchmark frameworks

### Key Data Structure
All time series use `MaskedTimeseries` with shape `(batch, variates, time_steps)`:
- Padding masks for valid vs. padded values
- ID masks for grouping related variates
- Timestamp information for temporal awareness

### Model Architecture
- Decoder-only transformer with alternating time-wise and space-wise attention
- Patch embedding converts time series into tokens
- Student-T mixture distributions for probabilistic outputs
- KV caching for efficient autoregressive inference

## Development Patterns

### Model Loading
```python
from model.toto import Toto
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
toto.compile()  # JIT compilation recommended
```

### Forecasting
```python
from inference.forecaster import TotoForecaster
forecaster = TotoForecaster(toto.model)
forecast = forecaster.forecast(inputs, prediction_length=336, num_samples=256)
```

### Testing Configuration
- Tests organized by component in `toto/test/`
- GPU tests marked with `@pytest.mark.cuda`
- Configuration in `pytest.ini` and `mypy.ini`

## Important Notes

- Model emphasizes median/quantile predictions over mean for better accuracy
- Designed for high-dimensional time series (tested on thousands of variates)
- Memory-efficient attention (xFormers) recommended for large models
- Supports variable prediction horizons and context lengths
- Zero-shot forecasting capability without fine-tuning