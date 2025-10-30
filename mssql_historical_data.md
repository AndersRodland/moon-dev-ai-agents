# MS SQL Forex Historical Data Documentation

## Overview

This document provides comprehensive information about the forex historical data stored in the MS SQL Server database. This data is being integrated into the AI trading system to enable forex backtesting alongside cryptocurrency trading strategies.

**Database Name:** TraderGenie
**Total Tables:** 610 DATA tables
**Total Forex Pairs:** 67 unique currency pairs
**Data Range:** January 2005 - September 2025 (~20+ years)
**Library Used:** `pymssql` (Python MS SQL Server client)

---

## Database Connection

### Environment Variables Required

The following environment variables must be set in `.env`:

```env
MSSQL_HOST=<server_hostname>
MSSQL_PORT=<port_number>         # Default: 1433
MSSQL_DATABASE=TraderGenie
MSSQL_USER=<username>
MSSQL_PASSWORD=<password>
MSSQL_DRIVER=<driver_name>       # Optional
MSSQL_TRUST_CERT=<true/false>    # Optional
MSSQL_ENCRYPT=<true/false>       # Optional
```

### Python Connection Code

```python
import pymssql
import os
from dotenv import load_dotenv

load_dotenv()

conn = pymssql.connect(
    server=os.getenv('MSSQL_HOST'),
    port=os.getenv('MSSQL_PORT') or '1433',
    database=os.getenv('MSSQL_DATABASE'),
    user=os.getenv('MSSQL_USER'),
    password=os.getenv('MSSQL_PASSWORD')
)
```

---

## Table Structure and Naming Convention

### Naming Pattern

All forex data tables follow this naming convention:

```
dbo.DATA_<BASE>_<QUOTE>_<TIMEFRAME>
```

**Examples:**
- `dbo.DATA_EUR_USD_M1` - EUR/USD 1-minute data
- `dbo.DATA_GBP_JPY_H4` - GBP/JPY 4-hour data
- `dbo.DATA_USD_CAD_D` - USD/CAD daily data

### Timeframe Codes

| Code | Description | Minutes |
|------|-------------|---------|
| M1   | 1 minute    | 1       |
| M3   | 3 minutes   | 3       |
| M5   | 5 minutes   | 5       |
| M10  | 10 minutes  | 10      |
| M15  | 15 minutes  | 15      |
| H1   | 1 hour      | 60      |
| H2   | 2 hours     | 120     |
| H4   | 4 hours     | 240     |
| D    | Daily       | 1440    |

### Column Schema

Every table has **14 columns** with the following structure:

| Column Name | Data Type      | Nullable | Description |
|-------------|----------------|----------|-------------|
| candleTime  | smalldatetime  | NO       | Timestamp of the candle (primary time column) |
| mid_o       | float          | NO       | Mid price open (average of bid/ask) |
| mid_h       | float          | NO       | Mid price high |
| mid_l       | float          | NO       | Mid price low |
| mid_c       | float          | NO       | Mid price close |
| volume      | int            | NO       | Tick volume (number of price changes) |
| bid_o       | float          | YES      | Bid price open |
| bid_h       | float          | YES      | Bid price high |
| bid_l       | float          | YES      | Bid price low |
| bid_c       | float          | YES      | Bid price close |
| ask_o       | float          | YES      | Ask price open |
| ask_h       | float          | YES      | Ask price high |
| ask_l       | float          | YES      | Ask price low |
| ask_c       | float          | YES      | Ask price close |

**Important Notes:**
- `candleTime` is the primary timestamp column (NOT 'time' or 'timestamp')
- Mid prices are ALWAYS populated (NOT NULL)
- Bid/Ask prices may be NULL in some records
- Volume represents tick volume (price changes), not actual traded volume

---

## Available Forex Pairs (67 Total)

### Major Pairs (8)
```
EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD
EUR_GBP
```

### Cross Pairs (59)

#### EUR Crosses (14)
```
EUR_AUD, EUR_CAD, EUR_CHF, EUR_CZK, EUR_DKK, EUR_GBP, EUR_HKD,
EUR_HUF, EUR_JPY, EUR_NOK, EUR_NZD, EUR_PLN, EUR_SEK, EUR_SGD,
EUR_TRY, EUR_ZAR
```

#### GBP Crosses (9)
```
GBP_AUD, GBP_CAD, GBP_CHF, GBP_HKD, GBP_JPY, GBP_NZD, GBP_PLN,
GBP_SGD, GBP_ZAR
```

#### USD Crosses (13)
```
USD_CAD, USD_CHF, USD_CZK, USD_DKK, USD_HKD, USD_HUF, USD_JPY,
USD_MXN, USD_NOK, USD_PLN, USD_SEK, USD_SGD, USD_THB, USD_TRY,
USD_ZAR
```

#### AUD Crosses (7)
```
AUD_CAD, AUD_CHF, AUD_HKD, AUD_JPY, AUD_NZD, AUD_SGD, AUD_USD
```

#### NZD Crosses (6)
```
NZD_CAD, NZD_CHF, NZD_HKD, NZD_JPY, NZD_SGD, NZD_USD
```

#### CAD Crosses (4)
```
CAD_CHF, CAD_HKD, CAD_JPY, CAD_SGD
```

#### Other Crosses (8)
```
CHF_HKD, CHF_JPY, CHF_ZAR, HKD_JPY, SGD_CHF, SGD_JPY,
TRY_JPY, ZAR_JPY
```

**All 67 pairs are available in all 9 timeframes** (610 total tables = 67 pairs × 9 timeframes + 7 miscellaneous tables)

---

## Data Quality Assessment

### EUR_USD_M1 Analysis (Representative Sample)

| Metric | Value |
|--------|-------|
| **Earliest Date** | 2005-01-02 18:29:00 |
| **Latest Date** | 2025-09-22 14:42:00 |
| **Time Span** | 7,567 days (20.72 years) |
| **Total Records** | 7,554,045 candles |
| **Duplicate Timestamps** | 0 |
| **Average Candles/Day** | ~998 (expected: 1,440 for 1-minute data) |

### Key Observations

1. **Data Completeness**: ~69% coverage for M1 data (998/1440 candles per day on average)
   - Likely due to weekend closures (forex markets closed Saturday-Sunday)
   - Potentially low-volume periods excluded

2. **Data Integrity**:
   - NO duplicate timestamps found
   - Clean, well-structured data
   - Consistent schema across all tables

3. **Price Data**:
   - Bid/Ask spreads are preserved in separate columns
   - Mid prices calculated as (bid + ask) / 2
   - All prices are float values with appropriate precision

4. **Volume Data**:
   - Tick volume (not actual traded volume)
   - Useful for volatility analysis and market activity

---

## Querying Examples

### Basic Data Retrieval

```python
import pymssql
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Connect
conn = pymssql.connect(
    server=os.getenv('MSSQL_HOST'),
    port=os.getenv('MSSQL_PORT') or '1433',
    database=os.getenv('MSSQL_DATABASE'),
    user=os.getenv('MSSQL_USER'),
    password=os.getenv('MSSQL_PASSWORD')
)

# Query EUR/USD 1-hour data for last 100 candles
query = """
SELECT TOP 100
    candleTime,
    mid_o, mid_h, mid_l, mid_c,
    volume,
    bid_c, ask_c
FROM dbo.DATA_EUR_USD_H1
ORDER BY candleTime DESC
"""

df = pd.read_sql(query, conn)
print(df.head())

conn.close()
```

### Get Data for Specific Date Range

```python
query = """
SELECT
    candleTime,
    mid_o, mid_h, mid_l, mid_c,
    volume
FROM dbo.DATA_GBP_USD_M15
WHERE candleTime >= '2024-01-01'
  AND candleTime < '2024-02-01'
ORDER BY candleTime ASC
"""

df = pd.read_sql(query, conn)
```

### Get Data with Bid/Ask Spreads

```python
query = """
SELECT
    candleTime,
    mid_c as close_price,
    bid_c,
    ask_c,
    (ask_c - bid_c) as spread,
    volume
FROM dbo.DATA_EUR_USD_M5
WHERE candleTime >= '2025-01-01'
ORDER BY candleTime DESC
"""

df = pd.read_sql(query, conn)
```

### Aggregate to Higher Timeframes

```python
# Convert M1 to M30 data
query = """
SELECT
    DATEADD(MINUTE,
        DATEDIFF(MINUTE, 0, candleTime) / 30 * 30,
        0) as candle_30m,
    MIN(mid_o) as open_30m,
    MAX(mid_h) as high_30m,
    MIN(mid_l) as low_30m,
    MAX(mid_c) as close_30m,
    SUM(volume) as volume_30m
FROM dbo.DATA_EUR_USD_M1
WHERE candleTime >= '2025-01-01'
GROUP BY DATEADD(MINUTE,
    DATEDIFF(MINUTE, 0, candleTime) / 30 * 30,
    0)
ORDER BY candle_30m DESC
"""

df = pd.read_sql(query, conn)
```

---

## Integration with Backtesting

### Using with backtesting.py Library

```python
import pymssql
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta

# Function to load forex data from MS SQL
def load_forex_data(pair='EUR_USD', timeframe='H1', start_date='2024-01-01', end_date='2025-01-01'):
    """
    Load forex data from MS SQL database for backtesting.

    Args:
        pair: Currency pair (e.g., 'EUR_USD', 'GBP_JPY')
        timeframe: Timeframe code (M1, M5, M15, H1, H4, D)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    conn = pymssql.connect(
        server=os.getenv('MSSQL_HOST'),
        port=os.getenv('MSSQL_PORT') or '1433',
        database=os.getenv('MSSQL_DATABASE'),
        user=os.getenv('MSSQL_USER'),
        password=os.getenv('MSSQL_PASSWORD')
    )

    table_name = f"dbo.DATA_{pair}_{timeframe}"

    query = f"""
    SELECT
        candleTime,
        mid_o as Open,
        mid_h as High,
        mid_l as Low,
        mid_c as Close,
        volume as Volume
    FROM {table_name}
    WHERE candleTime >= '{start_date}'
      AND candleTime < '{end_date}'
    ORDER BY candleTime ASC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Set candleTime as index
    df.set_index('candleTime', inplace=True)

    return df


# Example strategy
class ForexStrategy(Strategy):
    def init(self):
        # Add indicators using pandas_ta
        close = pd.Series(self.data.Close, index=self.data.index)
        self.sma20 = self.I(ta.sma, close, length=20)
        self.sma50 = self.I(ta.sma, close, length=50)

    def next(self):
        if crossover(self.sma20, self.sma50):
            self.buy()
        elif crossover(self.sma50, self.sma20):
            self.position.close()


# Load data and run backtest
data = load_forex_data(pair='EUR_USD', timeframe='H4', start_date='2024-01-01')
bt = Backtest(data, ForexStrategy, cash=10000, commission=0.0002)
stats = bt.run()
print(stats)
bt.plot()
```

### Using with pandas_ta Indicators

```python
import pandas_ta as ta

# Load data
df = load_forex_data('GBP_USD', 'M15', '2025-01-01', '2025-02-01')

# Add technical indicators
df.ta.sma(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(length=20, append=True)
df.ta.atr(length=14, append=True)

print(df.tail())
```

### Using with TA-Lib

```python
import talib

# Load data
df = load_forex_data('USD_JPY', 'H1', '2024-01-01', '2025-01-01')

# Calculate indicators
df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
    df['Close'],
    fastperiod=12,
    slowperiod=26,
    signalperiod=9
)

print(df.tail())
```

---

## Utility Functions

### Database Helper Class

```python
import pymssql
import pandas as pd
import os
from datetime import datetime

class ForexDataManager:
    """Utility class for managing forex data from MS SQL database."""

    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        """Establish database connection."""
        self.conn = pymssql.connect(
            server=os.getenv('MSSQL_HOST'),
            port=os.getenv('MSSQL_PORT') or '1433',
            database=os.getenv('MSSQL_DATABASE'),
            user=os.getenv('MSSQL_USER'),
            password=os.getenv('MSSQL_PASSWORD')
        )

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def get_available_pairs(self):
        """Get list of all available forex pairs."""
        query = """
        SELECT DISTINCT
            SUBSTRING(TABLE_NAME, 6,
                LEN(TABLE_NAME) - CHARINDEX('_', REVERSE(TABLE_NAME)) - 5
            ) as pair
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME LIKE 'DATA_%'
        ORDER BY pair
        """
        df = pd.read_sql(query, self.conn)
        return df['pair'].tolist()

    def get_available_timeframes(self, pair):
        """Get available timeframes for a specific pair."""
        query = f"""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME LIKE 'DATA_{pair}_%'
        ORDER BY TABLE_NAME
        """
        df = pd.read_sql(query, self.conn)
        timeframes = [
            table.split('_')[-1]
            for table in df['TABLE_NAME']
        ]
        return timeframes

    def get_data_range(self, pair, timeframe):
        """Get date range for a specific pair/timeframe."""
        table = f"dbo.DATA_{pair}_{timeframe}"
        query = f"""
        SELECT
            MIN(candleTime) as start_date,
            MAX(candleTime) as end_date,
            COUNT(*) as total_candles
        FROM {table}
        """
        df = pd.read_sql(query, self.conn)
        return df.iloc[0].to_dict()

    def load_data(self, pair, timeframe, start_date=None, end_date=None,
                  use_bid_ask=False, limit=None):
        """
        Load forex data with flexible options.

        Args:
            pair: Currency pair (e.g., 'EUR_USD')
            timeframe: Timeframe code (M1, M5, H1, etc.)
            start_date: Optional start date filter
            end_date: Optional end date filter
            use_bid_ask: If True, return bid/ask instead of mid prices
            limit: Optional limit on number of rows

        Returns:
            DataFrame with OHLCV data
        """
        table = f"dbo.DATA_{pair}_{timeframe}"

        if use_bid_ask:
            price_cols = "bid_o as Open, bid_h as High, bid_l as Low, bid_c as Close"
        else:
            price_cols = "mid_o as Open, mid_h as High, mid_l as Low, mid_c as Close"

        query = f"""
        SELECT {'TOP ' + str(limit) if limit else ''}
            candleTime,
            {price_cols},
            volume as Volume
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

        df = pd.read_sql(query, self.conn)
        df.set_index('candleTime', inplace=True)

        return df

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Usage example
with ForexDataManager() as fdm:
    # Get available pairs
    pairs = fdm.get_available_pairs()
    print(f"Available pairs: {len(pairs)}")

    # Get timeframes for EUR_USD
    timeframes = fdm.get_available_timeframes('EUR_USD')
    print(f"EUR_USD timeframes: {timeframes}")

    # Get date range
    date_range = fdm.get_data_range('EUR_USD', 'H1')
    print(f"Date range: {date_range}")

    # Load data
    df = fdm.load_data('EUR_USD', 'H4', start_date='2024-01-01', limit=100)
    print(df.head())
```

### Timeframe Conversion Utilities

```python
def timeframe_to_minutes(timeframe):
    """Convert timeframe code to minutes."""
    mapping = {
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
    return mapping.get(timeframe)


def minutes_to_timeframe(minutes):
    """Convert minutes to timeframe code."""
    mapping = {
        1: 'M1',
        3: 'M3',
        5: 'M5',
        10: 'M10',
        15: 'M15',
        60: 'H1',
        120: 'H2',
        240: 'H4',
        1440: 'D'
    }
    return mapping.get(minutes)


def parse_pair(pair_string):
    """
    Parse a pair string into base and quote currencies.

    Examples:
        'EUR_USD' -> ('EUR', 'USD')
        'EURUSD' -> ('EUR', 'USD')
        'EUR/USD' -> ('EUR', 'USD')
    """
    # Remove common separators
    clean = pair_string.replace('/', '').replace('_', '').replace('-', '')

    # Assume 3-letter currency codes
    if len(clean) == 6:
        return clean[:3], clean[3:]

    raise ValueError(f"Cannot parse pair: {pair_string}")


def format_pair_for_table(base, quote):
    """Format base and quote currencies for table name."""
    return f"{base.upper()}_{quote.upper()}"
```

---

## Performance Considerations

### Data Volume by Timeframe (Approximate)

| Timeframe | Records/Year (per pair) | Storage Size |
|-----------|------------------------|--------------|
| M1        | ~370,000               | ~15 MB       |
| M5        | ~74,000                | ~3 MB        |
| M15       | ~24,700                | ~1 MB        |
| H1        | ~6,200                 | ~250 KB      |
| H4        | ~1,550                 | ~62 KB       |
| D         | ~260                   | ~10 KB       |

**Total Database Size (Estimated):** ~40-50 GB for all 610 tables with 20 years of data

### Query Optimization Tips

1. **Always use WHERE clause with date filters** to limit data scanned
2. **Use TOP N** for recent data queries instead of LIMIT
3. **Select only needed columns** (avoid SELECT *)
4. **Use indexed candleTime column** for filtering
5. **Consider caching frequently used datasets** in memory or local files

### Example Optimized Query

```python
# GOOD - Selective and efficient
query = """
SELECT TOP 1000
    candleTime,
    mid_c as Close,
    volume
FROM dbo.DATA_EUR_USD_H1
WHERE candleTime >= '2025-01-01'
ORDER BY candleTime DESC
"""

# BAD - Scans entire table
query = """
SELECT *
FROM dbo.DATA_EUR_USD_M1
ORDER BY candleTime DESC
"""
```

---

## Integration Roadmap

### Phase 1: Data Access Layer (Current)
- ✅ Database connection established
- ✅ Table structure documented
- ✅ Basic query examples provided
- ✅ Test script (`test_mssql.py`) created

### Phase 2: Utility Functions (Next)
- Create `src/forex_data.py` with ForexDataManager class
- Add to `src/nice_funcs.py` for unified access
- Implement caching for frequently accessed data
- Add data validation and error handling

### Phase 3: Backtesting Integration
- Extend RBI agent to support forex data source
- Create forex-specific strategy templates
- Add pair correlation analysis
- Enable multi-asset backtests (crypto + forex)

### Phase 4: Agent Integration
- Create `forex_agent.py` for dedicated forex analysis
- Update `trading_agent.py` to support forex pairs
- Add forex-specific indicators and patterns
- Implement forex risk management (pip-based sizing)

### Phase 5: Live Trading (Future)
- Integrate with forex broker APIs (OANDA, Interactive Brokers, etc.)
- Implement forex-specific order types
- Add spread cost analysis
- Enable automated forex trading

---

## Common Issues and Solutions

### Issue: Connection Timeout

```python
# Solution: Add timeout parameter
conn = pymssql.connect(
    server=os.getenv('MSSQL_HOST'),
    port=os.getenv('MSSQL_PORT') or '1433',
    database=os.getenv('MSSQL_DATABASE'),
    user=os.getenv('MSSQL_USER'),
    password=os.getenv('MSSQL_PASSWORD'),
    timeout=30,  # Add timeout
    login_timeout=15
)
```

### Issue: Large Query Memory Usage

```python
# Solution: Use chunked reading
chunk_size = 10000
for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
    process_chunk(chunk)  # Process in batches
```

### Issue: Date Timezone Confusion

```python
# Solution: Always work in UTC
import pytz

df['candleTime'] = pd.to_datetime(df['candleTime']).dt.tz_localize('UTC')
```

### Issue: Pandas SQLAlchemy Warning

The warning about SQLAlchemy is cosmetic and can be ignored. For production, consider using SQLAlchemy:

```python
from sqlalchemy import create_engine

# Alternative connection using SQLAlchemy
connection_string = (
    f"mssql+pymssql://{os.getenv('MSSQL_USER')}:"
    f"{os.getenv('MSSQL_PASSWORD')}@"
    f"{os.getenv('MSSQL_HOST')}:{os.getenv('MSSQL_PORT')}/"
    f"{os.getenv('MSSQL_DATABASE')}"
)
engine = create_engine(connection_string)
df = pd.read_sql(query, engine)
```

---

## Testing and Validation

### Quick Validation Checklist

Run `test_mssql.py` to verify:
- ✅ Database connection successful
- ✅ All 67 forex pairs accessible
- ✅ All 9 timeframes available per pair
- ✅ No duplicate timestamps
- ✅ Data spans 20+ years
- ✅ Bid/ask spreads reasonable

### Data Quality Checks

```python
def validate_forex_data(df):
    """Validate forex data quality."""
    checks = {
        'no_nulls_in_mid': df[['Open', 'High', 'Low', 'Close']].isnull().sum().sum() == 0,
        'high_gte_low': (df['High'] >= df['Low']).all(),
        'high_gte_open_close': ((df['High'] >= df['Open']) & (df['High'] >= df['Close'])).all(),
        'low_lte_open_close': ((df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])).all(),
        'no_negative_prices': (df[['Open', 'High', 'Low', 'Close']] > 0).all().all(),
        'sorted_by_time': df.index.is_monotonic_increasing
    }

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")

    return all(checks.values())

# Usage
df = load_forex_data('EUR_USD', 'H1', '2024-01-01', '2024-02-01')
is_valid = validate_forex_data(df)
```

---

## Resources and References

### Test Script
- **Location:** `D:\dev\moon-dev-ai-agents\test_mssql.py`
- **Purpose:** Database exploration and validation
- **Run:** `python test_mssql.py`

### Environment Configuration
- **Location:** `D:\dev\moon-dev-ai-agents\.env`
- **Required Variables:** MSSQL_HOST, MSSQL_PORT, MSSQL_DATABASE, MSSQL_USER, MSSQL_PASSWORD

### Dependencies
- `pymssql==2.3.8` - MS SQL Server connector
- `pandas` - Data manipulation
- `python-dotenv` - Environment variable management

### Related Documentation
- [CLAUDE.md](./CLAUDE.md) - Main project documentation
- [src/models/README.md](./src/models/README.md) - LLM integration guide
- Backtesting.py docs: https://kernc.github.io/backtesting.py/

### External Resources
- pymssql documentation: https://pymssql.readthedocs.io/
- Forex market hours: https://www.forex.com/en-us/trading-academy/courses/introduction-to-forex/forex-market-hours/
- Currency pair naming conventions: https://www.investopedia.com/terms/c/currencypair.asp

---

## Summary

This MS SQL database provides **20+ years of high-quality forex historical data** for 67 currency pairs across 9 timeframes. The data structure is clean, consistent, and ready for integration with the AI trading system.

**Key Takeaways:**
- 610 total tables (67 pairs × 9 timeframes)
- Data from January 2005 to September 2025
- 14 columns per table (candleTime + OHLCV with bid/ask/mid)
- Zero duplicate timestamps
- Excellent for backtesting forex strategies
- Easy integration with backtesting.py, pandas_ta, and TA-Lib

**Next Steps:**
1. Create utility functions in `src/forex_data.py`
2. Integrate with RBI agent for forex backtesting
3. Enable forex analysis in existing agents
4. Develop forex-specific trading strategies

---

*Last Updated: 2025-10-29*
*Database: TraderGenie*
*Total Records: ~100M+ candles across all pairs and timeframes*
