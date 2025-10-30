"""
Test forex data integration through forex_data module.
"""

from src.forex_data import load_forex_data, get_forex_pairs, get_forex_timeframes

# Alias for compatibility
get_forex_data = load_forex_data

print("=" * 80)
print("FOREX INTEGRATION TEST via nice_funcs.py")
print("=" * 80)

# Test 1: Get pairs
print("\n[TEST 1] Getting available pairs...")
pairs = get_forex_pairs()
print(f"Found {len(pairs)} pairs")
print(f"Sample: {pairs[:10]}")

# Test 2: Get timeframes
print("\n[TEST 2] Getting EUR_USD timeframes...")
tfs = get_forex_timeframes('EUR_USD')
print(f"Timeframes: {tfs}")

# Test 3: Load data
print("\n[TEST 3] Loading EUR_USD H4 data (last 50 candles)...")
df = get_forex_data('EUR_USD', 'H4', limit=50)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Test 4: Load with date range
print("\n[TEST 4] Loading GBP_JPY daily data for 2024...")
df2 = get_forex_data('GBP_JPY', 'D', start_date='2024-01-01', end_date='2025-01-01')
print(f"Shape: {df2.shape}")
print(f"Date range: {df2.index.min()} to {df2.index.max()}")

# Test 5: Multiple pairs
print("\n[TEST 5] Loading multiple major pairs...")
majors = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']
for pair in majors:
    df_test = get_forex_data(pair, 'H1', limit=5)
    print(f"{pair}: {len(df_test)} rows, latest close: {df_test['Close'].iloc[-1]:.5f}")

print("\n" + "=" * 80)
print("INTEGRATION TEST COMPLETED SUCCESSFULLY")
print("=" * 80)
