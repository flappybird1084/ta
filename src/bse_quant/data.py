"""Loading the price files.

Each CSV in the data folder holds one asset, with columns
Date, Close/Last, Volume, Open, High, Low (prices written like "$28.43").
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLUMN_NAMES = {"Close/Last": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"}


@dataclass(frozen=True)
class MarketData:
    """Daily bars for every asset. Each field is a table: rows = dates, columns = assets."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def assets(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def returns(self) -> pd.DataFrame:
        """Close-to-close daily returns (0.01 = +1%)."""
        return self.close.pct_change()


def parse_price(text) -> float:
    return float(str(text).strip().replace("$", "").replace(",", ""))


def load_asset_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.rename(columns=COLUMN_NAMES)
    df["date"] = pd.to_datetime(raw["Date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].map(parse_price)
    return df.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]]


def asset_name(path: Path) -> str:
    """'Asset_A.csv' -> 'A'."""
    return path.stem.removeprefix("Asset_")


def load_market_data(folder) -> MarketData:
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {Path(folder).resolve()}")
    per_asset = {asset_name(f): load_asset_csv(f) for f in files}
    shared_dates = sorted(set.intersection(*(set(df.index) for df in per_asset.values())))
    fields = {
        field: pd.DataFrame({name: df.loc[shared_dates, field] for name, df in per_asset.items()})
        for field in ["open", "high", "low", "close", "volume"]
    }
    return MarketData(**fields)
