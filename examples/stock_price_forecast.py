import argparse
import pandas as pd
import torch

from data.util.dataset import MaskedTimeseries
from inference.forecaster import TotoForecaster
from model.toto import Toto


def load_data(csv_path: str) -> MaskedTimeseries:
    """Load closing price data from a CSV file."""
    df = pd.read_csv(csv_path, index_col=0)
    # assume columns represent different stocks and rows represent time steps
    values = torch.tensor(df.values.T, dtype=torch.get_default_dtype())

    padding_mask = torch.ones_like(values, dtype=torch.bool)
    id_mask = torch.zeros_like(values, dtype=torch.int)

    # optional timestamp information (not currently used by the model)
    timestamp_seconds = torch.arange(values.shape[-1]).expand(values.shape[0], -1)
    time_interval_seconds = torch.ones(values.shape[0], dtype=torch.int)

    return MaskedTimeseries(
        series=values,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_data(args.csv)

    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
    toto.to(device)
    toto.compile()

    forecaster = TotoForecaster(toto.model)
    forecast = forecaster.forecast(
        data,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
        samples_per_batch=args.samples_per_batch,
    )

    median = forecast.median
    print("Median forecast shape:", median.shape)
    if forecast.samples is not None:
        print("Samples shape:", forecast.samples.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast stock prices using Toto")
    parser.add_argument("csv", help="CSV file with stock closing prices")
    parser.add_argument("--prediction-length", type=int, default=30, help="Number of future steps to predict")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of probabilistic samples")
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=256,
        help="Number of samples processed per batch during inference",
    )
    main(parser.parse_args())
