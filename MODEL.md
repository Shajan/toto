# Toto Model Architecture

Toto is a decoder-only transformer designed for multivariate time series forecasting with these key architectural features:

## Model Input

The Toto model takes a `MaskedTimeseries` object as input, which contains:

**1. inputs** (`series`):
- Shape: `(batch, variates, series_len)`  
- The actual numerical time series values

**2. input_padding_mask** (`padding_mask`):
- Shape: `(batch, variates, series_len)`
- Boolean mask where `True` = valid data, `False` = padding
- Handles variable-length sequences

**3. id_mask** (`id_mask`):
- Shape: `(batch, variates, series_len)` or `(batch, variates, 1)`
- Integer IDs grouping related variates for multivariate attention
- Variates with same ID can attend to each other in space-wise layers

**4. Timestamps** (`timestamp_seconds`):
- Shape: `(batch, variates, series_len)`
- POSIX timestamps in seconds for temporal awareness

**5. Time Intervals** (`time_interval_seconds`):
- Shape: `(batch, variates)`
- Sampling frequency of each variate in seconds

## Core Architecture

**Patch-Based Processing**: Converts time series into overlapping patches (default patch_size=16, stride=8) for efficient processing

**Alternating Attention Pattern**: 
- **Time-wise attention**: Learns temporal dependencies within each variate
- **Space-wise attention**: Learns cross-variate dependencies
- Configurable ratio (e.g., 3 time-wise layers per 1 space-wise layer)

**Pre-norm Transformer**: LayerNorm applied before each sublayer (attention + feedforward) rather than after

## Key Components

**Embedding Layer**: PatchEmbedding converts time series patches into latent representations

**Transformer Stack**: Multiple layers alternating between:
- TimeWiseMultiheadAttention (causal, with rotary position encoding)  
- SpaceWiseMultiheadAttention (non-causal, for cross-variate relationships)

**Feed-Forward Networks**: SwiGLU activation (Swish + Gated Linear Unit) for improved performance

**Output Distribution**: Student-T mixture models for probabilistic forecasting with uncertainty estimation

## Specialized Features

**Scaling**: Multiple scaler types including CausalPatchStdMeanScaler for handling distribution shifts

**Memory Efficiency**: Optional xFormers integration for memory-efficient attention

**KV Caching**: Efficient autoregressive decoding during inference

**Zero-shot Capability**: No fine-tuning required for new time series domains

The model automatically handles padding to make series length divisible by patch stride, and applies scaling before patch embedding and transformer processing. It outputs probabilistic distributions enabling median/quantile forecasts with uncertainty bounds.