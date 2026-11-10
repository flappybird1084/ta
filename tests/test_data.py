import pandas as pd
import pytest

from bse_quant.data import load_market_data, parse_price


def write_csv(path, rows):
    header = "Date,Close/Last,Volume,Open,High,Low\n"
    path.write_text(header + "\n".join(rows) + "\n")


def test_parse_price_strips_dollar_sign():
    assert parse_price("$28.43") == pytest.approx(28.43)
    assert parse_price("28.43") == pytest.approx(28.43)
    assert parse_price(" $1,234.5 ") == pytest.approx(1234.5)


def test_load_market_data_reads_and_sorts(tmp_path):
    write_csv(
        tmp_path / "Asset_A.csv",
        [
            "2020-01-03,$11.00,200,$10.50,$11.20,$10.40",
            "2020-01-02,$10.00,100,$9.90,$10.10,$9.80",
        ],
    )
    write_csv(
        tmp_path / "Asset_B.csv",
        [
            "2020-01-02,$20.00,300,$19.00,$20.50,$18.50",
            "2020-01-03,$21.00,400,$20.00,$21.50,$19.50",
        ],
    )
    data = load_market_data(tmp_path)

    assert data.assets == ["A", "B"]
    assert list(data.dates) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert data.close.loc["2020-01-02", "A"] == pytest.approx(10.0)
    assert data.open.loc["2020-01-03", "A"] == pytest.approx(10.5)
    assert data.high.loc["2020-01-03", "B"] == pytest.approx(21.5)
    assert data.low.loc["2020-01-02", "B"] == pytest.approx(18.5)
    assert data.volume.loc["2020-01-03", "B"] == pytest.approx(400)


def test_load_market_data_keeps_only_shared_dates(tmp_path):
    write_csv(
        tmp_path / "Asset_A.csv",
        ["2020-01-02,$1,1,$1,$1,$1", "2020-01-03,$1,1,$1,$1,$1"],
    )
    write_csv(tmp_path / "Asset_B.csv", ["2020-01-03,$2,1,$2,$2,$2"])
    data = load_market_data(tmp_path)
    assert list(data.dates) == [pd.Timestamp("2020-01-03")]


def test_load_market_data_errors_on_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_market_data(tmp_path)


def test_daily_returns(tmp_path):
    write_csv(
        tmp_path / "Asset_A.csv",
        ["2020-01-02,$10,1,$10,$10,$10", "2020-01-03,$11,1,$11,$11,$11"],
    )
    data = load_market_data(tmp_path)
    assert data.returns().loc["2020-01-03", "A"] == pytest.approx(0.1)
