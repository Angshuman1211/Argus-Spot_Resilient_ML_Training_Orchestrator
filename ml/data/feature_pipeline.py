import pandas as pd
import numpy as np
import os
import glob

# Approximate On-Demand prices for your eu-north-1 region (in USD/hr)
# The ML model uses this to understand if Spot is currently cheap or expensive relatively.
OD_PRICES = {
    "m5.xlarge": 0.204,
    "m5.2xlarge": 0.408,
    "c5.xlarge": 0.188,
    "c5.2xlarge": 0.376,
    "g4dn.xlarge": 0.605,
}

def normalize_raw_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts raw prices from either source and returns one schema.

    fetch_spot_prices.py writes `availability_zone` / `product_desc`; the Lambda price
    collector (docs/A2_lambda_pipeline.md) writes `az` / `product`. Types without a known
    On-Demand price are dropped with a warning: the normalized_ratio feature would
    otherwise divide by a made-up default and silently feed the model a wrong value.
    """
    df = df.rename(columns={"az": "availability_zone", "product": "product_desc"})
    missing = {"timestamp", "instance_type", "availability_zone", "spot_price"} - set(df.columns)
    if missing:
        raise ValueError(f"raw prices are missing columns: {sorted(missing)}")
    df = df.drop_duplicates(["timestamp", "instance_type", "availability_zone"])
    unknown = sorted(set(df["instance_type"]) - set(OD_PRICES))
    if unknown:
        print(f"WARNING: dropping instance types with no On-Demand price in OD_PRICES: {unknown}")
        df = df[df["instance_type"].isin(OD_PRICES)]
    return df


def process_ts_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw spot prices for a specific instance type + AZ, 
    resamples them to fixed 5-minute intervals, and engineers features.
    """
    instance_type = group["instance_type"].iloc[0]
    az = group["availability_zone"].iloc[0]
    
    # Needs to be sorted by time
    group = group.sort_values("timestamp").set_index("timestamp")
    
    # 1. Resample to strict 5-minute intervals (Spot prices are event-driven, 
    # but Transformers need fixed timesteps). Forward-fill missing intervals.
    group = group.resample("5min").ffill()
    group = group.bfill() # Just in case the first row was NaN
    
    # Ensure categorical data is persisted after resample
    group["instance_type"] = instance_type
    group["availability_zone"] = az
    
    # 2. Price delta (change between consecutive 5-min samples)
    group["price_delta"] = group["spot_price"].diff().fillna(0)
    
    # 3. Rolling stats (15-min=3 steps, 1-hr=12 steps, 6-hr=72 steps)
    group["rolling_mean_15m"] = group["spot_price"].rolling(window=3, min_periods=1).mean()
    group["rolling_std_15m"]  = group["spot_price"].rolling(window=3, min_periods=1).std().fillna(0)
    
    group["rolling_mean_1h"]  = group["spot_price"].rolling(window=12, min_periods=1).mean()
    group["rolling_std_1h"]   = group["spot_price"].rolling(window=12, min_periods=1).std().fillna(0)
    
    group["rolling_mean_6h"]  = group["spot_price"].rolling(window=72, min_periods=1).mean()
    group["rolling_std_6h"]   = group["spot_price"].rolling(window=72, min_periods=1).std().fillna(0)
    
    # 4. Normalized ratio relative to On-Demand price
    od_price = OD_PRICES[instance_type]
    group["normalized_ratio"] = group["spot_price"] / od_price
    
    # 5. Temporal encodings (Cyclical time features using Sine/Cosine)
    # Time of day
    sec_in_day = 24 * 60 * 60
    time_seconds = group.index.hour * 3600 + group.index.minute * 60 + group.index.second
    group["sin_time_day"] = np.sin(2 * np.pi * time_seconds / sec_in_day)
    group["cos_time_day"] = np.cos(2 * np.pi * time_seconds / sec_in_day)
    
    # Day of week (0-6)
    group["sin_day_week"] = np.sin(2 * np.pi * group.index.dayofweek / 7)
    group["cos_day_week"] = np.cos(2 * np.pi * group.index.dayofweek / 7)
    
    return group.reset_index()

def load_interruption_rates(labels_dir: str, region: str = "eu-north-1", os_name: str = "Linux") -> pd.DataFrame:
    """
    Real interruption-frequency data published by AWS's Spot Instance Advisor
    (pulled via pull_spot_advisor.py). Aggregate per (region, instance_type) -
    not a per-timestep event - but it's a real signal, unlike the price-spike
    proxy label the model is trained against.
    """
    files = sorted(glob.glob(os.path.join(labels_dir, "spot_interruption_labels_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No spot_interruption_labels_*.csv in {labels_dir}. "
            f"Run: python ml/data/pull_spot_advisor.py --region {region}"
        )
    latest = files[-1]
    print(f"Loading interruption rates from {latest}...")
    labels = pd.read_csv(latest)
    labels = labels[(labels["region"] == region) & (labels["os"] == os_name)]
    return labels[["instance_type", "interruption_rate_max_pct"]].drop_duplicates("instance_type")

def build_features(input_csv: str, output_csv: str, labels_dir: str = None):
    print(f"Loading raw data from {input_csv}...")
    df = normalize_raw_prices(pd.read_csv(input_csv, parse_dates=["timestamp"]))
    
    print("Generating feature matrix...")
    processed_groups = []
    
    # Process each instance type and AZ individually so their time series don't mix
    for (itype, az), group in df.groupby(["instance_type", "availability_zone"]):
        processed_df = process_ts_group(group)
        processed_groups.append(processed_df)
        
    final_df = pd.concat(processed_groups, ignore_index=True)

    # Cross-AZ signal: how does this AZ's price compare to its sibling AZs for the
    # SAME instance type at the SAME timestamp? A single AZ pricing away from its
    # siblings is a classic precursor to an AZ-specific capacity crunch / reclaim -
    # a signal the previous feature set couldn't see at all, since every feature was
    # computed within one (instance_type, AZ) series in isolation.
    az_stats = (
        final_df.groupby(["instance_type", "timestamp"])["spot_price"]
        .mean()
        .rename("instance_type_az_mean_price")
        .reset_index()
    )
    final_df = final_df.merge(az_stats, on=["instance_type", "timestamp"], how="left")
    final_df["az_price_divergence"] = (
        (final_df["spot_price"] - final_df["instance_type_az_mean_price"])
        / final_df["instance_type_az_mean_price"].replace(0, np.nan)
    ).fillna(0)
    final_df = final_df.drop(columns=["instance_type_az_mean_price"])

    # Real interruption rate (AWS Spot Instance Advisor), per instance type. This is
    # the same value for every row of a given instance type - a prior, not a
    # time-varying signal - but it's real AWS data instead of the price-spike proxy.
    if labels_dir is not None:
        rates = load_interruption_rates(labels_dir)
        final_df = final_df.merge(rates, on="instance_type", how="left")
        final_df = final_df.rename(columns={"interruption_rate_max_pct": "instance_interruption_rate"})
        final_df["instance_interruption_rate"] = final_df["instance_interruption_rate"].fillna(0) / 100.0

    # Drop rows where we don't have enough data (just clean up)
    final_df = final_df.dropna()
    
    print(f"Shape of engineered feature dataset: {final_df.shape}")
    print("\nSample engineered features:")
    print(final_df[["timestamp", "spot_price", "price_delta", "rolling_mean_1h", "normalized_ratio", "sin_time_day"]].head())
    
    final_df.to_csv(output_csv, index=False)
    print(f"\nSaved engineered features to {output_csv}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(base_dir, "raw_spot_prices.csv")
    out_path = os.path.join(base_dir, "features.csv")
    labels_dir = os.path.join(base_dir, "labels")

    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found. Run fetch_spot_prices.py first.")
        exit(1)

    build_features(in_path, out_path, labels_dir=labels_dir)