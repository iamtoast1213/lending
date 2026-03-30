"""
Fetch Deribit Level 2 orderbook data from CoinMetrics API.

Fetches market orderbook snapshots with configurable depth (100 levels or full book).
Uses monthly chunking for memory efficiency and resume capability.
"""

import argparse
import glob
import os
from datetime import datetime

import pandas as pd
from coinmetrics.api_client import CoinMetricsClient
from dateutil.relativedelta import relativedelta


def get_existing_months(chunks_dir: str) -> set:
    """Get set of (year, month) tuples for existing chunk files."""
    existing = set()
    if not os.path.exists(chunks_dir):
        return existing

    # Look for files like 2022/2022-01.csv or directly 2022-01.csv
    for year_dir in glob.glob(os.path.join(chunks_dir, "*")):
        if os.path.isdir(year_dir):
            for csv_file in glob.glob(os.path.join(year_dir, "*.csv")):
                basename = os.path.basename(csv_file)
                # Parse YYYY-MM.csv format
                try:
                    parts = basename.replace(".csv", "").split("-")
                    if len(parts) >= 2:
                        year, month = int(parts[0]), int(parts[1])
                        existing.add((year, month))
                except (ValueError, IndexError):
                    pass
    return existing


def organize_chunks_into_years(chunks_dir: str):
    """Move chunk files from flat directory into year subdirectories."""
    if not os.path.exists(chunks_dir):
        return

    for csv_file in glob.glob(os.path.join(chunks_dir, "*.csv")):
        basename = os.path.basename(csv_file)
        # Parse YYYY-MM... format to get year
        try:
            year = basename.split("-")[0]
            if year.isdigit() and len(year) == 4:
                year_dir = os.path.join(chunks_dir, year)
                os.makedirs(year_dir, exist_ok=True)
                dest_path = os.path.join(year_dir, basename)
                if not os.path.exists(dest_path):
                    os.rename(csv_file, dest_path)
        except (ValueError, IndexError):
            pass


def combine_all_chunks(
    chunks_dir: str,
    output_path: str,
    dedup_cols: list,
    filter_perpetuals: bool = False,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> int:
    """Combine monthly chunk CSVs within the date range into a single parquet file.

    Reads CSV chunks, deduplicates on the given column list
    (e.g. ["coin_metrics_id", "market"]), and writes a single parquet file.
    Drops the unnamed index column written by CoinMetrics export_to_csv.
    Only includes chunks whose YYYY-MM falls within [start_dt, end_dt).
    """
    # Collect all monthly chunk paths in chronological order
    all_chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "*", "*.csv")))

    # Filter to only chunks within the requested date range
    chunk_files = []
    for path in all_chunk_files:
        basename = os.path.basename(path).replace(".csv", "")
        try:
            parts = basename.split("-")
            year, month = int(parts[0]), int(parts[1])
            chunk_date = datetime(year, month, 1)
            if start_dt and chunk_date < datetime(start_dt.year, start_dt.month, 1):
                continue
            if end_dt and chunk_date >= datetime(
                end_dt.year, end_dt.month, 1
            ) + relativedelta(months=1):
                continue
        except (ValueError, IndexError):
            pass  # include files that don't match YYYY-MM pattern
        chunk_files.append(path)
    if not chunk_files:
        print(f"    No chunk data found in {chunks_dir}")
        return 0

    seen = set()
    processed_chunks = []

    for chunk_path in chunk_files:
        try:
            df = pd.read_csv(chunk_path)
        except Exception as e:
            print(f"    Warning: Could not read {chunk_path}: {e}")
            continue

        if len(df) == 0:
            continue

        # Drop unnamed index column from export_to_csv
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        # Verify dedup columns exist
        missing = [c for c in dedup_cols if c not in df.columns]
        if missing:
            print(
                f"    Warning: {chunk_path} missing columns {missing}, skipping dedup"
            )
        else:
            # Build composite keys and filter to unseen rows
            keys = list(zip(*(df[c] for c in dedup_cols)))
            mask = []
            for key in keys:
                if key in seen:
                    mask.append(False)
                else:
                    seen.add(key)
                    mask.append(True)
            df = df[mask]

        # Filter perpetuals if requested
        if filter_perpetuals and "market" in df.columns:
            orig_len = len(df)
            df = df[~df["market"].str.contains("PERPETUAL", case=False, na=False)]
            filtered = orig_len - len(df)
            if filtered > 0:
                print(
                    f"    Filtered {filtered} perpetuals from {os.path.basename(chunk_path)}"
                )

        if len(df) == 0:
            continue

        # Sort chunk by time
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="mixed")
            df = df.sort_values("time")

        chunk_name = os.path.basename(chunk_path)
        print(f"    {chunk_name}: {len(df)} records")
        processed_chunks.append(df)

    if not processed_chunks:
        return 0

    combined = pd.concat(processed_chunks, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    return len(combined)


def fetch_orderbooks(
    client: CoinMetricsClient,
    asset: str,
    market_type: str,
    start_time: str,
    end_time: str,
    granularity: str,
    depth_limit: str,
    output_path: str,
    resume: bool = True,
) -> int:
    """Fetch orderbook data using wildcard pattern with monthly chunking.

    Fetches data one month at a time, saving to yearly subdirectories,
    then combines into the final output file.

    If resume=True, only fetches months that don't already have chunk files.

    Args:
        client: CoinMetrics API client
        asset: Asset symbol (btc, eth)
        market_type: Market type (option, future)
        start_time: Start time ISO string
        end_time: End time ISO string
        granularity: Data granularity (raw, 1m, 1h, 1d)
        depth_limit: Orderbook depth ("100" or "full_book")
        output_path: Path for final combined parquet file
        resume: Whether to resume from existing chunks
    """
    pattern = f"deribit-{asset.upper()}-*-{market_type}"
    print(f"  Fetching {market_type} orderbooks: {pattern}")
    print(f"  Depth limit: {depth_limit}, Granularity: {granularity}")

    # Set up chunks directory (e.g., data/btc_option_orderbooks/)
    chunks_dir = output_path.replace(".parquet", "")
    os.makedirs(chunks_dir, exist_ok=True)

    # Organize any existing flat chunks into year directories
    organize_chunks_into_years(chunks_dir)

    # Determine which months need to be fetched
    # Strip timezone for internal date arithmetic (API calls use original strings)
    start_dt = pd.to_datetime(start_time).tz_localize(None)
    end_dt = pd.to_datetime(end_time).tz_localize(None)

    existing_months = get_existing_months(chunks_dir) if resume else set()
    if existing_months:
        print(f"  Found {len(existing_months)} existing monthly chunks")

    # Generate list of months to fetch
    months_to_fetch = []
    current = datetime(start_dt.year, start_dt.month, 1)
    end_boundary = datetime(end_dt.year, end_dt.month, 1) + relativedelta(months=1)

    while current < end_boundary:
        if (current.year, current.month) not in existing_months:
            months_to_fetch.append(current)
        current += relativedelta(months=1)

    if not months_to_fetch:
        print(f"  All months already fetched, combining chunks...")
        filter_perpetuals = market_type == "future"
        total = combine_all_chunks(
            chunks_dir,
            output_path,
            ["coin_metrics_id", "market"],
            filter_perpetuals,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        print(f"  -> {total} records")
        return total

    print(
        f"  Fetching {len(months_to_fetch)} months: {months_to_fetch[0].strftime('%Y-%m')} to {months_to_fetch[-1].strftime('%Y-%m')}"
    )

    # Adaptive parameters based on depth_limit
    # Orderbook data is more voluminous than quotes
    if depth_limit == "full_book":
        page_size = 1000
        time_increment = relativedelta(days=3)
    else:
        page_size = 5000
        time_increment = relativedelta(days=5)

    # Fetch each missing month
    for month_start in months_to_fetch:
        month_end = month_start + relativedelta(months=1)

        # Clamp to actual date range
        fetch_start = max(month_start, start_dt)
        fetch_end = min(month_end, end_dt)

        if fetch_start >= fetch_end:
            continue

        year_dir = os.path.join(chunks_dir, str(month_start.year))
        os.makedirs(year_dir, exist_ok=True)
        chunk_path = os.path.join(year_dir, f"{month_start.strftime('%Y-%m')}.csv")

        print(
            f"    {month_start.strftime('%Y-%m')}: {fetch_start.strftime('%Y-%m-%d')} to {fetch_end.strftime('%Y-%m-%d')}...",
            end=" ",
            flush=True,
        )

        try:
            client.get_market_orderbooks(
                markets=pattern,
                start_time=fetch_start.isoformat(),
                end_time=fetch_end.isoformat(),
                granularity=granularity,
                depth_limit=depth_limit,
                page_size=page_size,
                end_inclusive=False,
            ).parallel(time_increment=time_increment).export_to_csv(chunk_path)

            # Count records in chunk
            chunk_df = pd.read_csv(chunk_path)
            print(f"{len(chunk_df)} records")
        except Exception as e:
            print(f"ERROR: {e}")
            # Don't raise - continue with next month
            continue

    # Combine all chunks into final output
    print(f"  Combining chunks into {output_path}...")
    filter_perpetuals = market_type == "future"
    total = combine_all_chunks(
        chunks_dir,
        output_path,
        ["coin_metrics_id", "market"],
        filter_perpetuals,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    print(f"  -> {total} records")
    return total


def fetch_reference_rates(
    client: CoinMetricsClient,
    asset: str,
    start_time: str,
    end_time: str,
    frequency: str,
    output_path: str,
    resume: bool = True,
) -> int:
    """Fetch reference rates (spot prices) with monthly chunking.

    Fetches ReferenceRateUSD one month at a time, saving to yearly
    subdirectories, then combines into the final output file.
    """
    print(f"  Fetching reference rates for {asset}")

    chunks_dir = output_path.replace(".csv", "")
    os.makedirs(chunks_dir, exist_ok=True)
    organize_chunks_into_years(chunks_dir)

    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    existing_months = get_existing_months(chunks_dir) if resume else set()
    if existing_months:
        print(f"  Found {len(existing_months)} existing monthly chunks")

    months_to_fetch = []
    current = datetime(start_dt.year, start_dt.month, 1)
    end_boundary = datetime(end_dt.year, end_dt.month, 1) + relativedelta(months=1)

    while current < end_boundary:
        if (current.year, current.month) not in existing_months:
            months_to_fetch.append(current)
        current += relativedelta(months=1)

    if not months_to_fetch:
        print(f"  All months already fetched, combining chunks...")
        total = combine_all_chunks(chunks_dir, output_path, ["asset", "time"])
        print(f"  -> {total} records")
        return total

    print(
        f"  Fetching {len(months_to_fetch)} months: "
        f"{months_to_fetch[0].strftime('%Y-%m')} to {months_to_fetch[-1].strftime('%Y-%m')}"
    )

    for month_start in months_to_fetch:
        month_end = month_start + relativedelta(months=1)
        fetch_start = max(month_start, start_dt)
        fetch_end = min(month_end, end_dt)

        if fetch_start >= fetch_end:
            continue

        year_dir = os.path.join(chunks_dir, str(month_start.year))
        os.makedirs(year_dir, exist_ok=True)
        chunk_path = os.path.join(year_dir, f"{month_start.strftime('%Y-%m')}.csv")

        print(
            f"    {month_start.strftime('%Y-%m')}: "
            f"{fetch_start.strftime('%Y-%m-%d')} to {fetch_end.strftime('%Y-%m-%d')}...",
            end=" ",
            flush=True,
        )

        try:
            client.get_asset_metrics(
                assets=[asset],
                metrics=["ReferenceRateUSD"],
                start_time=fetch_start.isoformat(),
                end_time=fetch_end.isoformat(),
                frequency=frequency,
                page_size=10000,
                end_inclusive=False,
            ).parallel(time_increment=relativedelta(days=7)).export_to_csv(chunk_path)

            chunk_df = pd.read_csv(chunk_path)
            print(f"{len(chunk_df)} records")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print(f"  Combining chunks into {output_path}...")
    total = combine_all_chunks(chunks_dir, output_path, ["asset", "time"])
    print(f"  -> {total} records")
    return total


def fetch_data(
    api_key: str,
    asset: str = "btc",
    market_type: str = "option",
    start_time: str = "2022-01-01",
    end_time: str = "2026-01-01",
    granularity: str = "1h",
    depth_limit: str = "10",
    output_dir: str = "./data",
    resume: bool = True,
) -> dict:
    """Fetch orderbook data for specified asset and market type.

    Args:
        api_key: CoinMetrics API key
        asset: Asset symbol (btc, eth)
        market_type: Market type (option, future, both)
        start_time: Start time ISO string
        end_time: End time ISO string
        granularity: Data granularity (raw, 1m, 1h, 1d)
        depth_limit: Orderbook depth ("100" or "full_book")
        output_dir: Output directory for parquet files
        resume: Whether to resume from existing data
    """
    os.makedirs(output_dir, exist_ok=True)

    client = CoinMetricsClient(api_key)

    print(f"\n{'=' * 60}")
    print(f"Fetching CoinMetrics orderbook data for {asset.upper()}")
    print(f"Market type: {market_type}")
    print(f"Date range: {start_time} to {end_time}")
    print(f"Granularity: {granularity}, Depth: {depth_limit}")
    print(f"Resume mode: {'enabled' if resume else 'disabled'}")
    print(f"{'=' * 60}\n")

    results = {}
    market_types = ["option", "future"] if market_type == "both" else [market_type]

    for mtype in market_types:
        output_path = os.path.join(output_dir, f"{asset}_{mtype}_orderbooks.parquet")
        count = fetch_orderbooks(
            client,
            asset,
            mtype,
            start_time,
            end_time,
            granularity,
            depth_limit,
            output_path,
            resume=resume,
        )
        results[f"{mtype}_orderbooks"] = count
        print()

    # Fetch reference rates (spot prices) at hourly granularity
    rates_path = os.path.join(output_dir, f"{asset}_reference_rates.csv")
    rates_count = fetch_reference_rates(
        client, asset, start_time, end_time, "1h", rates_path, resume=resume
    )
    results["reference_rates"] = rates_count

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    for mtype in market_types:
        output_path = os.path.join(output_dir, f"{asset}_{mtype}_orderbooks.parquet")
        print(f"  {output_path}: {results[f'{mtype}_orderbooks']} records")
    print(f"  {rates_path}: {results['reference_rates']} records")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Deribit Level 2 orderbook data from CoinMetrics API"
    )
    parser.add_argument(
        "--asset",
        default="btc",
        help="Asset to fetch (default: btc)",
    )
    parser.add_argument(
        "--market-type",
        default="option",
        choices=["option", "future", "both"],
        help="Market type to fetch (default: option)",
    )
    parser.add_argument(
        "--start-time",
        default="2022-01-01",
        help="Start time (default: 2022-01-01)",
    )
    parser.add_argument(
        "--end-time",
        default="2026-01-01",
        help="End time (default: 2026-01-01)",
    )
    parser.add_argument(
        "--granularity",
        default="1h",
        choices=["raw", "1m", "1h", "1d"],
        help="Data granularity (default: 1h)",
    )
    parser.add_argument(
        "--depth-limit",
        default="10",
        choices=["10", "100", "full_book"],
        help="Orderbook depth limit (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing data (default: enabled)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume, fetch fresh data",
    )
    args = parser.parse_args()

    API_KEY = os.environ.get("CM_API_KEY")
    if not API_KEY:
        print("Error: Set CM_API_KEY environment variable")
        exit(1)

    resume = args.resume and not args.no_resume

    result = fetch_data(
        api_key=API_KEY,
        asset=args.asset,
        market_type=args.market_type,
        start_time=args.start_time,
        end_time=args.end_time,
        granularity=args.granularity,
        depth_limit=args.depth_limit,
        output_dir=args.output_dir,
        resume=resume,
    )
