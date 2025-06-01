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
#                                      ^     ^
#                                     CPU  Memory
#                                    every every  
#                                     60s  120s

# Time grid (minutes):     0    1    2    3    4
# CPU data (variate 0): 5 samples (every minute) - all positions valid
inputs[:, 0, :] = [cpu_t0, cpu_t1, cpu_t2, cpu_t3, cpu_t4]
padding_mask[:, 0, :] = [True, True, True, True, True]

# Memory data (variate 1): 3 samples (every 2 minutes at positions 0, 2, 4)  
inputs[:, 1, :] = [mem_t0, 0, mem_t2, 0, mem_t4]
padding_mask[:, 1, :] = [True, False, True, False, True]
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

Toto is a decoder-only transformer that processes time series through three key innovations:

### 1. Patch-Based Processing
Instead of processing individual time steps, the model groups consecutive points into overlapping patches:
- **patch_size=16**: Each patch contains 16 time steps
- **stride=8**: New patch every 8 steps (50% overlap)
- **Result**: 32 time steps become 3 patches, reducing sequence length 8x

This makes attention computation much more efficient while preserving temporal information through overlapping patches.

### 2. Dual Attention System
The transformer alternates between two types of layers:
- **TIME layers**: Learn temporal patterns (how each variate evolves over time)
- **SPACE layers**: Learn cross-variate relationships (how different metrics relate at the same time)

The ratio is configurable - for example, 3 TIME layers followed by 1 SPACE layer, repeating throughout the stack.

### 3. Pre-norm Architecture with Residual Connections
Each transformer layer uses a pre-norm design with residual connections:
- **Pre-norm**: LayerNorm applied before each sublayer (rather than after)
- **Residual connections**: Skip connections around both attention and feedforward sublayers
- **Pattern**: `Input → LayerNorm → Attention → Add → LayerNorm → MLP → Add → Output`

The residual connections enable:
- **Gradient flow**: Direct paths for gradients during backpropagation
- **Training stability**: Enables training of deep transformer stacks
- **Feature preservation**: Lower-level features can combine with higher-level representations

**Key Insight**: By alternating between temporal and cross-variate attention, the model learns both how individual metrics change over time and how different metrics influence each other - essential for multivariate forecasting.

## Key Components

**Embedding Layer**: PatchEmbedding converts time series patches into latent representations

**Transformer Stack**: Multiple layers alternating between two attention types:

### Layer Types vs Attention Heads

**Layer Types** (what the pattern `[TIME, TIME, TIME, SPACE, ...]` refers to):
- **TIME layers**: Transformer layers that apply attention across the time dimension
- **SPACE layers**: Transformer layers that apply attention across the variate dimension  
- Each layer type determines which dimension the attention mechanism operates over

**Attention Heads** (different concept):
- Within each layer, there are multiple attention heads (controlled by `num_heads` parameter)
- All heads in a TIME layer attend across time steps
- All heads in a SPACE layer attend across variates
- Example: A TIME layer with `num_heads=8` has 8 attention heads, all attending temporally

### Time-wise Multi-Head Attention
- **Purpose**: Learns temporal dependencies within each variate
- **Mechanism**: Causal attention over the time dimension
- **Implementation**: Flattens variates along batch dimension, applies attention across time steps
- **Features**:
  - Causal masking (can only attend to past/current time steps)
  - Rotary position embeddings for relative temporal positioning
  - KV caching support for efficient autoregressive inference

### Space-wise Multi-Head Attention  
- **Purpose**: Learns cross-variate relationships at the same time point
- **Mechanism**: Bidirectional attention over the variate dimension
- **Implementation**: Flattens time steps along batch dimension, applies attention across variates
- **Features**:
  - Non-causal (can attend to all variates)
  - No rotary embeddings (invariant to variate ordering)
  - Controlled by `id_mask` for grouping related variates

### Attention Implementation Details
- **Memory Efficiency**: Optional xFormers memory-efficient attention for large sequences
- **Multi-head**: Each layer uses multiple attention heads (`num_heads` parameter)
- **Unified Architecture**: Both attention types share the same base implementation with different axis configurations

**Feed-Forward Networks**: SwiGLU activation (Swish + Gated Linear Unit) for improved performance

**Output Distribution**: Student-T mixture models for probabilistic forecasting with uncertainty estimation

## Specialized Features

**Scaling**: Multiple scaler types including CausalPatchStdMeanScaler for handling distribution shifts

**Memory Efficiency**: Optional xFormers integration for memory-efficient attention

**KV Caching**: Efficient autoregressive decoding during inference

**Zero-shot Capability**: No fine-tuning required for new time series domains

The model automatically handles padding to make series length divisible by patch stride, and applies scaling before patch embedding and transformer processing.

## Model Output

### Model Raw Output

The Toto model's `forward()` method returns a `TotoOutput` object containing:

1. **`distribution`**: A PyTorch Distribution object (e.g., Student-T or mixture) representing forecast uncertainty
2. **`loc`**: Location parameters for scaling back to original values - shape `(batch, variate)`  
3. **`scale`**: Scale parameters for scaling back to original values - shape `(batch, variate)`

### Forecaster Output

The `TotoForecaster` converts the raw model output into a user-friendly `Forecast` object:

1. **`mean`**: Mean predictions - shape `(batch, variate, future_time_steps)`
2. **`samples`**: Optional probabilistic samples - shape `(batch, variate, future_time_steps, samples)`

### Probabilistic Distributions

The model outputs probabilistic distributions rather than point predictions:

**Single Student-T Distribution**:
```python
distribution = torch.distributions.StudentT(
    df=tensor([[2.5, 3.1, 2.8]]),      # degrees of freedom  
    loc=tensor([[0.0, 0.1, -0.2]]),    # location parameter
    scale=tensor([[1.2, 0.8, 1.5]])    # scale parameter
)
```

**Mixture of Student-T Distributions**:
```python
distribution = torch.distributions.MixtureSameFamily(
    mixture_distribution=Categorical(...),   # mixing weights
    component_distribution=StudentT(...)     # k_components Student-T distributions
)
```

**Distribution Capabilities**:
- `distribution.sample()` - Generate random forecast samples
- `distribution.mean` - Get mean predictions
- `distribution.log_prob(x)` - Compute probability of observed values
- `distribution.cdf(x)` - Cumulative distribution function

### Example Output Structure

```python
# Forecast object for 2 variates, 3 future time steps
forecast = Forecast(
    mean=tensor([[0.5, 0.6, 0.7],      # variate 0 predictions (e.g., CPU)
                 [2.1, 2.3, 2.0]]),    # variate 1 predictions (e.g., Memory)  
    samples=tensor([[[...], [...], [...]], # 256 samples per prediction
                    [[...], [...], [...]]])
)

# Access specific predictions
cpu_predictions = forecast.mean[0, 0, :]  # batch 0, variate 0
memory_predictions = forecast.mean[0, 1, :]  # batch 0, variate 1
confidence_intervals = forecast.quantile([0.1, 0.9])  # 80% confidence interval
median_forecast = forecast.median  # Median predictions (often more accurate than mean)
```

**Key Benefits**: The probabilistic output enables uncertainty quantification, confidence intervals, risk assessment, and multiple scenario generation - essential for robust time series forecasting.
