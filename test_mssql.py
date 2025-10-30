"""
Test script for MS SQL Server forex historical data connection and exploration.

This script connects to the MS SQL database containing 60+ forex pairs with
multiple timeframes and explores the data structure, availability, and quality.
"""

import pymssql
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

# Database configuration from .env
MSSQL_HOST = os.getenv('MSSQL_HOST')
MSSQL_PORT = os.getenv('MSSQL_PORT')
MSSQL_DATABASE = os.getenv('MSSQL_DATABASE')
MSSQL_USER = os.getenv('MSSQL_USER')
MSSQL_PASSWORD = os.getenv('MSSQL_PASSWORD')


def get_connection():
    """Establish connection to MS SQL Server."""
    try:
        conn = pymssql.connect(
            server=MSSQL_HOST,
            port=MSSQL_PORT if MSSQL_PORT else '1433',
            database=MSSQL_DATABASE,
            user=MSSQL_USER,
            password=MSSQL_PASSWORD
        )
        print(f"[OK] Connected to database: {MSSQL_DATABASE}")
        return conn
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        raise


def get_table_structure(conn, table_name):
    """Get column information for a specific table."""
    query = f"""
    SELECT
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{table_name}'
    ORDER BY ORDINAL_POSITION
    """
    df = pd.read_sql(query, conn)
    return df


def get_sample_data(conn, table_name, limit=10):
    """Get sample data from a table."""
    query = f"SELECT TOP {limit} * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df


def get_data_date_range(conn, table_name, date_column='time'):
    """Get the date range of data in a table."""
    query = f"""
    SELECT
        MIN({date_column}) as earliest_date,
        MAX({date_column}) as latest_date,
        COUNT(*) as total_records
    FROM {table_name}
    """
    df = pd.read_sql(query, conn)
    return df


def get_all_data_tables(conn):
    """Find all DATA tables (forex pairs) in the database."""
    query = """
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    AND TABLE_NAME LIKE 'DATA_%'
    ORDER BY TABLE_NAME
    """
    df = pd.read_sql(query, conn)
    return df


def parse_table_name(table_name):
    """Parse forex pair and timeframe from table name.

    Expected format: DATA_EUR_USD_M1 or dbo.DATA_EUR_USD_M1
    Returns: {'pair': 'EUR_USD', 'timeframe': 'M1', 'base': 'EUR', 'quote': 'USD'}
    """
    # Remove dbo. prefix if present
    clean_name = table_name.replace('dbo.', '')

    # Split by underscore
    parts = clean_name.split('_')

    if len(parts) >= 4 and parts[0] == 'DATA':
        return {
            'pair': f"{parts[1]}_{parts[2]}",
            'base': parts[1],
            'quote': parts[2],
            'timeframe': parts[3] if len(parts) == 4 else '_'.join(parts[3:])
        }
    return None


def analyze_data_quality(conn, table_name, date_column='time'):
    """Analyze data quality - gaps, duplicates, etc."""
    # Check for duplicates
    dup_query = f"""
    SELECT {date_column}, COUNT(*) as count
    FROM {table_name}
    GROUP BY {date_column}
    HAVING COUNT(*) > 1
    """
    duplicates = pd.read_sql(dup_query, conn)

    # Get row count
    count_query = f"SELECT COUNT(*) as total FROM {table_name}"
    total_rows = pd.read_sql(count_query, conn)

    return {
        'total_rows': total_rows['total'].iloc[0],
        'duplicate_timestamps': len(duplicates),
        'duplicates_sample': duplicates.head(5) if len(duplicates) > 0 else None
    }


def main():
    """Main exploration function."""
    print("=" * 80)
    print("MS SQL FOREX DATA EXPLORATION")
    print("=" * 80)

    # Connect to database
    conn = get_connection()

    try:
        # 1. Explore EUR_USD_M1 table structure
        print("\n" + "=" * 80)
        print("1. EUR_USD_M1 TABLE STRUCTURE")
        print("=" * 80)

        target_table = "dbo.DATA_EUR_USD_M1"
        structure = get_table_structure(conn, "DATA_EUR_USD_M1")
        print("\nColumn Structure:")
        print(structure.to_string(index=False))

        # 2. Get sample data
        print("\n" + "=" * 80)
        print("2. SAMPLE DATA (First 10 rows)")
        print("=" * 80)

        sample = get_sample_data(conn, target_table, limit=10)
        print(f"\nShape: {sample.shape}")
        print(f"Columns: {list(sample.columns)}")
        print("\nData Preview:")
        print(sample.to_string(index=False))

        # 3. Date range analysis
        print("\n" + "=" * 80)
        print("3. DATA DATE RANGE")
        print("=" * 80)

        # Try common date column names
        date_columns = ['candleTime', 'time', 'timestamp', 'datetime', 'date', 'Time', 'DateTime']
        date_col_found = None

        for col in date_columns:
            if col in sample.columns:
                date_col_found = col
                break

        if date_col_found:
            date_range = get_data_date_range(conn, target_table, date_col_found)
            print(f"\nDate column: {date_col_found}")
            print(date_range.to_string(index=False))

            # Calculate time span
            earliest = pd.to_datetime(date_range['earliest_date'].iloc[0])
            latest = pd.to_datetime(date_range['latest_date'].iloc[0])
            time_span = latest - earliest
            print(f"\nTime span: {time_span.days} days ({time_span.days/365.25:.2f} years)")
        else:
            print(f"\nNo standard date column found. Available columns: {list(sample.columns)}")

        # 4. Discover all forex pairs
        print("\n" + "=" * 80)
        print("4. DISCOVERING ALL FOREX PAIRS")
        print("=" * 80)

        all_tables = get_all_data_tables(conn)
        print(f"\nTotal DATA tables found: {len(all_tables)}")

        # Parse and organize by pair
        pairs_dict = {}
        timeframes_set = set()

        for table in all_tables['TABLE_NAME']:
            parsed = parse_table_name(table)
            if parsed:
                pair = parsed['pair']
                timeframe = parsed['timeframe']

                if pair not in pairs_dict:
                    pairs_dict[pair] = []
                pairs_dict[pair].append(timeframe)
                timeframes_set.add(timeframe)

        print(f"\nUnique Forex Pairs: {len(pairs_dict)}")
        print(f"Unique Timeframes: {sorted(timeframes_set)}")

        print("\n--- Forex Pairs and Their Available Timeframes ---")
        for pair in sorted(pairs_dict.keys()):
            timeframes = sorted(pairs_dict[pair])
            print(f"{pair:12} -> {', '.join(timeframes)}")

        # 5. Data quality analysis
        if date_col_found:
            print("\n" + "=" * 80)
            print("5. DATA QUALITY ANALYSIS (EUR_USD_M1)")
            print("=" * 80)

            quality = analyze_data_quality(conn, target_table, date_col_found)
            print(f"\nTotal Records: {quality['total_rows']:,}")
            print(f"Duplicate Timestamps: {quality['duplicate_timestamps']}")

            if quality['duplicates_sample'] is not None and len(quality['duplicates_sample']) > 0:
                print("\nSample Duplicates:")
                print(quality['duplicates_sample'].to_string(index=False))

        # 6. Sample data from different timeframes
        print("\n" + "=" * 80)
        print("6. SAMPLING DIFFERENT TIMEFRAMES (EUR_USD)")
        print("=" * 80)

        test_timeframes = ['M1', 'M15', 'H1', 'H4', 'D']
        for tf in test_timeframes:
            table = f"dbo.DATA_EUR_USD_{tf}"
            try:
                sample_tf = get_sample_data(conn, table, limit=3)
                print(f"\n{tf} (Shape: {sample_tf.shape}):")
                print(sample_tf.head(3).to_string(index=False))
            except Exception as e:
                print(f"\n{tf}: Not available or error - {e}")

        print("\n" + "=" * 80)
        print("EXPLORATION COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Error during exploration: {e}")
        raise
    finally:
        conn.close()
        print("\n[OK] Database connection closed")


if __name__ == "__main__":
    main()
