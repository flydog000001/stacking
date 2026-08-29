from __future__ import annotations

"""Merge official no-poroelastic complete-pipeline Bootstrap members.

This is the aggregation stage of the fixed-candidate Bootstrap experiment. It
does not fit a model. It validates the 100 complete Stacking members for every
station-day, then summarizes their prediction distribution as final-Stacking
model uncertainty.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bootstrap_uncertainty" / "no_poroelastic_complete_pipeline"

# Four batches x 25 repetitions x 3 years x 11 outer folds = 132 files.
OFFSETS = {100, 125, 150, 175}
EXPECTED_FILE_COUNT = 132
EXPECTED_MEMBER_COUNT = 100
EXPECTED_STATION_COUNT = 70
EXPECTED_DAILY_RECORD_COUNT = 76650
KEYS = ["年份", "轮次", "站点", "day_index", "GNSS真实值"]
MEMBER_PREDICTION = "Bootstrap成员预测"


def official_member_files() -> list[Path]:
    """Return only the four official 25-member batches for every year/fold."""
    files = []
    pattern = "california_final_stacking_bootstrap_members_no_poroelastic_*.csv.gz"
    for path in OUT.glob(pattern):
        match = re.search(r"batch(\d+)\.csv\.gz$", path.name)
        if match and int(match.group(1)) in OFFSETS:
            files.append(path)
    files.sort()
    if len(files) != EXPECTED_FILE_COUNT:
        raise RuntimeError(
            f"Official member-file count is {len(files)}, expected {EXPECTED_FILE_COUNT}."
        )
    return files


def load_and_validate_members(files: list[Path]) -> tuple[pd.DataFrame, pd.Series]:
    """Load members and require exactly 100 values for every station-day."""
    members = pd.concat(
        [pd.read_csv(path, compression="gzip") for path in files],
        ignore_index=True,
    )
    required = set(KEYS + [MEMBER_PREDICTION])
    missing = required.difference(members.columns)
    if missing:
        raise KeyError(f"Member files lack required columns: {sorted(missing)}")

    counts = members.groupby(KEYS, dropna=False).size()
    invalid = counts[~counts.eq(EXPECTED_MEMBER_COUNT)]
    if not invalid.empty:
        raise RuntimeError(
            "Bootstrap member-count validation failed: "
            f"{invalid.value_counts().to_dict()}"
        )
    return members, counts


def summarize_daily(members: pd.DataFrame) -> pd.DataFrame:
    """Calculate member mean, standard deviation, and empirical quantiles."""
    daily = members.groupby(KEYS, as_index=False)[MEMBER_PREDICTION].agg(
        Bootstrap点预测="mean",
        Bootstrap标准差="std",
        BootstrapP2_5=lambda value: value.quantile(0.025),
        BootstrapP97_5=lambda value: value.quantile(0.975),
        Bootstrap次数="count",
    )
    if daily["站点"].nunique() != EXPECTED_STATION_COUNT:
        raise RuntimeError(
            f"Station coverage is {daily['站点'].nunique()}, expected {EXPECTED_STATION_COUNT}."
        )
    if len(daily) != EXPECTED_DAILY_RECORD_COUNT:
        raise RuntimeError(
            f"Daily-record count is {len(daily)}, expected {EXPECTED_DAILY_RECORD_COUNT}."
        )
    if not daily["Bootstrap次数"].eq(EXPECTED_MEMBER_COUNT).all():
        raise RuntimeError("Daily summary contains a non-100 Bootstrap member count.")
    return daily.sort_values(KEYS).reset_index(drop=True)


def uncertainty_statistics(group: pd.DataFrame) -> pd.Series:
    """Summarize model uncertainty only, without interval or residual metrics."""
    sd = group["Bootstrap标准差"].to_numpy(dtype=float)
    prediction = group["Bootstrap点预测"].to_numpy(dtype=float)
    truth = group["GNSS真实值"].to_numpy(dtype=float)
    return pd.Series({
        "样本数": len(group),
        "Bootstrap标准差均值": float(np.mean(sd)),
        "Bootstrap标准差中位数": float(np.median(sd)),
        "Bootstrap标准差P95": float(np.quantile(sd, 0.95)),
        "Bootstrap标准差最大值": float(np.max(sd)),
        "点预测RMSE": float(np.sqrt(np.mean((truth - prediction) ** 2))),
    })


def build_outputs(
    daily: pd.DataFrame, counts: pd.Series, file_count: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = uncertainty_statistics(daily).to_frame().T
    overall.insert(0, "全部站点数", daily["站点"].nunique())
    overall.insert(1, "年份数", daily["年份"].nunique())
    overall.insert(2, "外层折数", len(daily[["年份", "轮次"]].drop_duplicates()))
    overall.insert(3, "文件数", file_count)

    annual = (
        daily.groupby("年份", group_keys=False)
        .apply(uncertainty_statistics)
        .reset_index()
    )
    station_fold = (
        daily.groupby(["年份", "轮次", "站点"], group_keys=False)
        .apply(uncertainty_statistics)
        .reset_index()
    )
    zero_sd = daily["Bootstrap标准差"] <= 1e-12
    audit = pd.DataFrame([{
        "正式批次文件数": file_count,
        "逐日记录数": len(daily),
        "站点数": daily["站点"].nunique(),
        "成员数最小值": int(counts.min()),
        "成员数最大值": int(counts.max()),
        "零标准差记录数": int(zero_sd.sum()),
        "零标准差比例": float(zero_sd.mean()),
        "验证结果": "通过",
    }])
    return overall, annual, station_fold, audit


def main() -> None:
    files = official_member_files()
    members, counts = load_and_validate_members(files)
    daily = summarize_daily(members)
    overall, annual, station_fold, audit = build_outputs(daily, counts, len(files))

    daily_path = OUT / "california_final_stacking_bootstrap_daily_uncertainty_no_poroelastic.csv.gz"
    summary_path = OUT / "california_final_stacking_bootstrap_summary_no_poroelastic.xlsx"
    daily.to_csv(daily_path, index=False, compression="gzip", encoding="utf-8-sig")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="整体汇总", index=False)
        annual.to_excel(writer, sheet_name="逐年汇总", index=False)
        station_fold.to_excel(writer, sheet_name="逐站逐轮汇总", index=False)
        audit.to_excel(writer, sheet_name="完整性核验", index=False)

    print(overall.to_string(index=False))
    print(audit.to_string(index=False))
    print(f"Saved daily output: {daily_path}")
    print(f"Saved summary workbook: {summary_path}")


if __name__ == "__main__":
    main()
