# Toto Model Architecture

Toto is a decoder-only transformer designed for multivariate time series forecasting with these key architectural features:

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

The model processes shape `(batch, variates, time_steps)` and outputs probabilistic distributions enabling median/quantile forecasts with uncertainty bounds.