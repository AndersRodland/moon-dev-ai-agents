"""
Forex Data Manager - MS SQL Historical Data Integration

This module provides access to 67 forex pairs with 20+ years of historical data
across 9 timeframes (M1, M3, M5, M10, M15, H1, H2, H4, D).

Database: TraderGenie (MS SQL Server)
Total Tables: 610 (67 pairs × 9 timeframes)
Date Range: January 2005 - September 2025

Usage:
    from src.forex_data import ForexDataManager, load_forex_data

    # Quick load
    df = load_forex_data('EUR_USD', 'H1', start_date='2024-01-01')

    # Advanced usage
    with ForexDataManager() as fdm:
        pairs = fdm.get_available_pairs()
        df = fdm.load_data('GBP_USD', 'M15', limit=1000)
"""

import pymssql
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pickle

# Load environment variables
load_dotenv()


# Timeframe constants
TIMEFRAMES = {
    'M1': 1,
    'M3': 3,
    'M5': 5,
    'M10': 10,
    'M15': 15,
    'H1': 60,
    'H2': 120,
    'H4': 240,
    'D': 1440
}

# Cache directory
CACHE_DIR = Path(__file__).parent / 'data' / 'forex_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def timeframe_to_minutes(timeframe: str) -> Optional[int]:
    """
    Convert timeframe code to minutes.

    Args:
        timeframe: Timeframe code (M1, M5, H1, etc.)

    Returns:
        Number of minutes or None if invalid
    """
    return TIMEFRAMES.get(timeframe.upper())


def minutes_to_timeframe(minutes: int) -> Optional[str]:
    """
    Convert minutes to timeframe code.

    Args:
        minutes: Number of minutes

    Returns:
        Timeframe code or None if no match
    """
    reverse_map = {v: k for k, v in TIMEFRAMES.items()}
    return reverse_map.get(minutes)


def parse_pair(pair_string: str) -> Tuple[str, str]:
    """
    Parse a pair string into base and quote currencies.

    Args:
        pair_string: Pair in various formats (EUR_USD, EURUSD, EUR/USD)

    Returns:
        Tuple of (base, quote) currencies

    Examples:
        >>> parse_pair('EUR_USD')
        ('EUR', 'USD')
        >>> parse_pair('GBPJPY')
        ('GBP', 'JPY')
        >>> parse_pair('EUR/USD')
        ('EUR', 'USD')
    """
    # Remove common separators
    clean = pair_string.replace('/', '').replace('_', '').replace('-', '').upper()

    # Assume 3-letter currency codes
    if len(clean) == 6:
        return clean[:3], clean[3:]

    raise ValueError(f"Cannot parse pair: {pair_string}")


def format_pair_for_table(base: str, quote: str) -> str:
    """
    Format base and quote currencies for table name.

    Args:
        base: Base currency (e.g., 'EUR')
        quote: Quote currency (e.g., 'USD')

    Returns:
        Formatted pair string (e.g., 'EUR_USD')
    """
    return f"{base.upper()}_{quote.upper()}"


def validate_forex_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate forex data quality.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Dictionary of validation checks and results
    """
    required_cols = ['Open', 'High', 'Low', 'Close']

    checks = {
        'has_required_columns': all(col in df.columns for col in required_cols),
        'no_nulls_in_ohlc': df[required_cols].isnull().sum().sum() == 0,
        'high_gte_low': (df['High'] >= df['Low']).all(),
        'high_gte_open': (df['High'] >= df['Open']).all(),
        'high_gte_close': (df['High'] >= df['Close']).all(),
        'low_lte_open': (df['Low'] <= df['Open']).all(),
        'low_lte_close': (df['Low'] <= df['Close']).all(),
        'no_negative_prices': (df[required_cols] > 0).all().all(),
        'sorted_by_time': df.index.is_monotonic_increasing,
        'no_duplicates': not df.index.duplicated().any()
    }

    return checks


class ForexDataManager:
    """
    Utility class for managing forex data from MS SQL database.

    Provides methods to:
    - Connect to database
    - Query available pairs and timeframes
    - Load historical data with caching
    - Validate data quality

    Usage:
        with ForexDataManager() as fdm:
            df = fdm.load_data('EUR_USD', 'H1', start_date='2024-01-01')
            print(df.head())
    """

    def __init__(self, enable_cache: bool = True, cache_ttl_hours: int = 24):
        """
        Initialize ForexDataManager.

        Args:
            enable_cache: Enable local file caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.conn = None
        self.enable_cache = enable_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.connect()

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = pymssql.connect(
                server=os.getenv('MSSQL_HOST'),
                port=os.getenv('MSSQL_PORT') or '1433',
                database=os.getenv('MSSQL_DATABASE'),
                user=os.getenv('MSSQL_USER'),
                password=os.getenv('MSSQL_PASSWORD'),
                timeout=30,
                login_timeout=15
            )
            print(f"[ForexDataManager] Connected to {os.getenv('MSSQL_DATABASE')}")
        except Exception as e:
            print(f"[ForexDataManager] Connection failed: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("[ForexDataManager] Connection closed")

    def get_available_pairs(self) -> List[str]:
        """
        Get list of all available forex pairs.

        Returns:
            List of forex pair strings (e.g., ['EUR_USD', 'GBP_JPY', ...])
        """
        query = """
        SELECT DISTINCT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME LIKE 'DATA_%'
        ORDER BY TABLE_NAME
        """

        df = pd.read_sql(query, self.conn)

        # Extract unique pairs
        pairs = set()
        for table in df['TABLE_NAME']:
            parts = table.split('_')
            if len(parts) >= 4 and parts[0] == 'DATA':
                pair = f"{parts[1]}_{parts[2]}"
                pairs.add(pair)

        return sorted(list(pairs))

    def get_available_timeframes(self, pair: str) -> List[str]:
        """
        Get available timeframes for a specific pair.

        Args:
            pair: Forex pair (e.g., 'EUR_USD')

        Returns:
            List of timeframe codes (e.g., ['M1', 'M5', 'H1', 'D'])
        """
        # Clean pair format
        if '_' not in pair:
            base, quote = parse_pair(pair)
            pair = format_pair_for_table(base, quote)

        query = f"""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME LIKE 'DATA_{pair}_%'
        ORDER BY TABLE_NAME
        """

        try:
            df = pd.read_sql(query, self.conn)
            timeframes = [
                table.split('_')[-1]
                for table in df['TABLE_NAME']
            ]
            return timeframes
        except Exception as e:
            print(f"[ForexDataManager] Error getting timeframes for {pair}: {e}")
            return []

    def get_data_range(self, pair: str, timeframe: str) -> Dict:
        """
        Get date range and statistics for a specific pair/timeframe.

        Args:
            pair: Forex pair (e.g., 'EUR_USD')
            timeframe: Timeframe code (e.g., 'H1')

        Returns:
            Dictionary with start_date, end_date, and total_candles
        """
        # Clean pair format
        if '_' not in pair:
            base, quote = parse_pair(pair)
            pair = format_pair_for_table(base, quote)

        table = f"dbo.DATA_{pair}_{timeframe.upper()}"

        query = f"""
        SELECT
            MIN(candleTime) as start_date,
            MAX(candleTime) as end_date,
            COUNT(*) as total_candles
        FROM {table}
        """

        try:
            df = pd.read_sql(query, self.conn)
            result = df.iloc[0].to_dict()

            # Calculate time span
            if result['start_date'] and result['end_date']:
                start = pd.to_datetime(result['start_date'])
                end = pd.to_datetime(result['end_date'])
                result['time_span_days'] = (end - start).days
                result['time_span_years'] = result['time_span_days'] / 365.25

            return result
        except Exception as e:
            print(f"[ForexDataManager] Error getting data range: {e}")
            return {}

    def _get_cache_key(self, pair: str, timeframe: str, start_date: Optional[str],
                       end_date: Optional[str], use_bid_ask: bool) -> str:
        """Generate cache key for data request."""
        parts = [pair, timeframe, str(start_date), str(end_date), str(use_bid_ask)]
        return '_'.join(parts).replace(' ', '_').replace(':', '-')

    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        if not self.enable_cache:
            return None

        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        meta_file = CACHE_DIR / f"{cache_key}.meta.json"

        if not cache_file.exists() or not meta_file.exists():
            return None

        # Check if cache is expired
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            cache_time = datetime.fromisoformat(meta['cached_at'])
            if datetime.now() - cache_time > self.cache_ttl:
                return None

            # Load cached data
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)

            print(f"[ForexDataManager] Loaded from cache: {cache_key}")
            return df

        except Exception as e:
            print(f"[ForexDataManager] Cache read error: {e}")
            return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame):
        """Save data to cache."""
        if not self.enable_cache:
            return

        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        meta_file = CACHE_DIR / f"{cache_key}.meta.json"

        try:
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

            # Save metadata
            meta = {
                'cached_at': datetime.now().isoformat(),
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                }
            }

            with open(meta_file, 'w') as f:
                json.dump(meta, f)

            print(f"[ForexDataManager] Saved to cache: {cache_key}")

        except Exception as e:
            print(f"[ForexDataManager] Cache write error: {e}")

    def load_data(self, pair: str, timeframe: str,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  use_bid_ask: bool = False,
                  limit: Optional[int] = None,
                  validate: bool = True) -> pd.DataFrame:
        """
        Load forex data with flexible options.

        Args:
            pair: Currency pair (e.g., 'EUR_USD', 'EURUSD', 'EUR/USD')
            timeframe: Timeframe code (M1, M5, M15, H1, H4, D)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            use_bid_ask: If True, return bid prices; False returns mid prices
            limit: Optional limit on number of rows (most recent)
            validate: Run data quality validation

        Returns:
            DataFrame with OHLCV data indexed by candleTime

        Example:
            df = fdm.load_data('EUR_USD', 'H1', start_date='2024-01-01', limit=1000)
        """
        # Clean pair format
        if '_' not in pair:
            base, quote = parse_pair(pair)
            pair = format_pair_for_table(base, quote)

        timeframe = timeframe.upper()

        # Check cache
        cache_key = self._get_cache_key(pair, timeframe, start_date, end_date, use_bid_ask)
        cached_df = self._get_cached_data(cache_key)
        if cached_df is not None:
            if limit:
                return cached_df.tail(limit)
            return cached_df

        # Build query
        table = f"dbo.DATA_{pair}_{timeframe}"

        if use_bid_ask:
            price_cols = """
                bid_o as [Open],
                bid_h as [High],
                bid_l as [Low],
                bid_c as [Close],
                ask_o as [Ask_Open],
                ask_h as [Ask_High],
                ask_l as [Ask_Low],
                ask_c as [Ask_Close]
            """
        else:
            price_cols = """
                mid_o as [Open],
                mid_h as [High],
                mid_l as [Low],
                mid_c as [Close]
            """

        query = f"""
        SELECT {'TOP ' + str(limit) if limit else ''}
            candleTime,
            {price_cols},
            volume as [Volume]
        FROM {table}
        """

        conditions = []
        if start_date:
            conditions.append(f"candleTime >= '{start_date}'")
        if end_date:
            conditions.append(f"candleTime < '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY candleTime ASC"

        # Execute query
        try:
            df = pd.read_sql(query, self.conn)

            if df.empty:
                print(f"[ForexDataManager] No data found for {pair} {timeframe}")
                return df

            # Set index
            df.set_index('candleTime', inplace=True)

            # Validate data
            if validate:
                validation = validate_forex_data(df)
                failed_checks = [k for k, v in validation.items() if not v]
                if failed_checks:
                    print(f"[ForexDataManager] WARNING: Failed validation checks: {failed_checks}")

            # Cache the result
            if not limit:  # Only cache unlimited queries
                self._save_to_cache(cache_key, df)

            print(f"[ForexDataManager] Loaded {len(df)} rows for {pair} {timeframe}")
            return df

        except Exception as e:
            print(f"[ForexDataManager] Error loading data: {e}")
            raise

    def clear_cache(self, pair: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear cached data.

        Args:
            pair: Optional specific pair to clear (clears all if None)
            timeframe: Optional specific timeframe to clear
        """
        if pair is None and timeframe is None:
            # Clear all cache
            for file in CACHE_DIR.glob('*'):
                file.unlink()
            print("[ForexDataManager] Cleared all cache")
        else:
            # Clear specific cache files
            pattern = f"{pair or '*'}_{timeframe or '*'}*"
            for file in CACHE_DIR.glob(pattern):
                file.unlink()
            print(f"[ForexDataManager] Cleared cache for {pair} {timeframe}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function
def load_forex_data(pair: str, timeframe: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    use_bid_ask: bool = False,
                    limit: Optional[int] = None) -> pd.DataFrame:
    """
    Quick function to load forex data without context manager.

    Args:
        pair: Currency pair (e.g., 'EUR_USD', 'EURUSD')
        timeframe: Timeframe code (M1, M5, H1, etc.)
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        use_bid_ask: Return bid prices instead of mid
        limit: Optional row limit

    Returns:
        DataFrame with OHLCV data

    Example:
        df = load_forex_data('EUR_USD', 'H1', start_date='2024-01-01')
    """
    with ForexDataManager() as fdm:
        return fdm.load_data(pair, timeframe, start_date, end_date, use_bid_ask, limit)


def get_forex_pairs() -> List[str]:
    """
    Get list of all available forex pairs.

    Returns:
        List of forex pair strings
    """
    with ForexDataManager() as fdm:
        return fdm.get_available_pairs()


def get_forex_timeframes(pair: str) -> List[str]:
    """
    Get available timeframes for a pair.

    Args:
        pair: Forex pair

    Returns:
        List of timeframe codes
    """
    with ForexDataManager() as fdm:
        return fdm.get_available_timeframes(pair)


if __name__ == "__main__":
    """Test script for forex data module."""

    print("=" * 80)
    print("FOREX DATA MODULE TEST")
    print("=" * 80)

    # Test 1: Get available pairs
    print("\n[TEST 1] Getting available pairs...")
    pairs = get_forex_pairs()
    print(f"Found {len(pairs)} forex pairs")
    print(f"Sample pairs: {pairs[:5]}")

    # Test 2: Get timeframes for EUR_USD
    print("\n[TEST 2] Getting timeframes for EUR_USD...")
    timeframes = get_forex_timeframes('EUR_USD')
    print(f"Available timeframes: {timeframes}")

    # Test 3: Load data using convenience function
    print("\n[TEST 3] Loading EUR_USD H1 data (last 100 candles)...")
    df = load_forex_data('EUR_USD', 'H1', limit=100)
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")

    # Test 4: Load with date range
    print("\n[TEST 4] Loading GBP_USD M15 data for Jan 2024...")
    df2 = load_forex_data('GBP_USD', 'M15', start_date='2024-01-01', end_date='2024-02-01')
    print(f"Shape: {df2.shape}")
    print(f"Date range: {df2.index.min()} to {df2.index.max()}")

    # Test 5: Using context manager
    print("\n[TEST 5] Using ForexDataManager context manager...")
    with ForexDataManager() as fdm:
        info = fdm.get_data_range('EUR_USD', 'H1')
        print(f"EUR_USD H1 info: {info}")

        df3 = fdm.load_data('USD_JPY', 'D', limit=30)
        print(f"USD_JPY daily data: {df3.shape}")

    # Test 6: Data validation
    print("\n[TEST 6] Validating data...")
    validation = validate_forex_data(df)
    print("Validation results:")
    for check, result in validation.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
