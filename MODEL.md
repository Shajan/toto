# Toto Model Architecture

Toto is a decoder-only transformer designed for multivariate time series forecasting with these key architectural features:

## Model Input

The Toto model takes a `MaskedTimeseries` object as input, which contains:

**1. inputs** (`series`):
- Shape: `(batch, variates, series_len)`  
- The actual numerical time series values
- Purpose: Provides the core data that the model learns temporal and cross-variate patterns from

**2. input_padding_mask** (`padding_mask`):
- Shape: `(batch, variates, series_len)`
- Boolean mask where `True` = valid data, `False` = padding
- Purpose: Enables efficient batching of variable-length sequences by indicating which positions contain real data versus padding, ensuring the model doesn't learn from artificial padding values

**3. id_mask** (`id_mask`):
- Shape: `(batch, variates, series_len)` or `(batch, variates, 1)`
- Integer IDs grouping related variates for multivariate attention
- Purpose: Allows the model to learn cross-variate relationships within logical groups while preventing interference between unrelated time series. Creates block masks during space-wise attention so variates with the same ID can attend to each other

**4. Timestamps** (`timestamp_seconds`):
- Shape: `(batch, variates, series_len)`
- POSIX timestamps in seconds for temporal awareness
- Purpose: Provides absolute temporal context that helps the model understand seasonal patterns, time-of-day effects, and long-term trends beyond just relative sequence positions

**5. Time Intervals** (`time_interval_seconds`):
- Shape: `(batch, variates)`
- Sampling frequency of each variate in seconds
- Purpose: Informs the model about the temporal resolution of each variate, enabling proper handling of mixed-frequency data where different metrics are sampled at different rates

## Mixed-Frequency Support

The model supports variates with different sampling frequencies through the combination of `time_interval_seconds` and `input_padding_mask`:

- `time_interval_seconds` specifies the **sampling frequency** (interval between samples) for each variate
- `input_padding_mask` indicates which positions contain valid data vs. padding when variates are aligned to a common time grid
- The number of actual samples depends on the total time series length and alignment strategy

**Example**: Monitoring system with 1-minute CPU metrics and 2-minute memory metrics over 4 minutes
```python
# Sampling frequencies: CPU every 60s, Memory every 120s
time_interval_seconds = torch.tensor([[60, 120]])
#                                     ^    ^
#                                   CPU  Memory
#                                 every every  
#                                  60s  120s

# Time grid (minutes):     0    1    2    3    4
# CPU data (variate 0): 5 samples (every minute) - all positions valid
inputs[0, 0, :] = [cpu_0, cpu_1, cpu_2, cpu_3, cpu_4]
padding_mask[0, 0, :] = [True, True, True, True, True]

# Memory data (variate 1): 3 samples (every 2 minutes at positions 0, 2, 4)  
inputs[0, 1, :] = [mem_0, 0, mem_2, 0, mem_4]
padding_mask[0, 1, :] = [True, False, True, False, True]
```

This allows the model to learn appropriate temporal relationships despite different sampling rates, with attention mechanisms properly weighting available data while ignoring padded positions.

## Variate Grouping for Space-wise Attention

The `id_mask` enables grouping of related variates so that space-wise attention operates within logical groups:

**Example**: Monitoring two separate servers
```python
# 4 variates: CPU+Memory for Server A, CPU+Memory for Server B
id_mask = torch.tensor([[0, 0, 1, 1]])  # shape: (1, 4, 1)

# Server A: CPU and Memory can attend to each other
# Server B: CPU and Memory can attend to each other  
# But Server A metrics cannot attend to Server B metrics

# Results in block attention mask:
# Variate:  A_CPU  A_Mem  B_CPU  B_Mem
# A_CPU   [  ✓     ✓      ✗      ✗   ]
# A_Mem   [  ✓     ✓      ✗      ✗   ] 
# B_CPU   [  ✗     ✗      ✓      ✓   ]
# B_Mem   [  ✗     ✗      ✓      ✓   ]
```

This prevents interference between unrelated systems while allowing the model to learn cross-metric relationships within each logical group.

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