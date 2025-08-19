from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI  # Changed this import
#from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import logging
import sys
from datetime import datetime
import numpy as np
import os 
from clickhouse_driver import Client
import urllib.parse
import clickhouse_connect
from datetime import datetime, timedelta
import uuid



METABASE_QUERY_TEMPLATES = {
    "hourly_production": """
WITH device_lookup AS (
    SELECT virtual_device_id, device_name FROM devices
),
production_calculation_query AS (
    WITH hourly_windows AS (
        SELECT
            toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
            device_id,
            value,
            CASE
                WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 7 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '07:30-08:30'
                WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 8 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '07:30-08:30'
                -- Add other time slots as needed
                ELSE NULL
            END AS hour_bucket,
            toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
        FROM device_metrics
        WHERE parameter = 'length'
            AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('{start_date} 07:30:00', 'Asia/Colombo')
            AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('{end_date} 19:30:00', 'Asia/Colombo')
            {device_filter}
    ),
    production_windows AS (
        SELECT
            device_id,
            hour_bucket,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS production_output
        FROM hourly_windows
        WHERE hour_bucket IS NOT NULL
        GROUP BY device_id, hour_bucket, date
    )
)
SELECT 
    dl.device_name AS machine_name,
    pcq.hour_bucket AS time_slot,
    pcq.date AS production_date,
    pcq.production_output AS hourly_production
FROM production_calculation_query pcq
LEFT JOIN device_lookup dl ON pcq.device_id = dl.virtual_device_id
ORDER BY pcq.date, pcq.hour_bucket, dl.device_name
""",

    "daily_production": """
WITH device_lookup AS (
    SELECT virtual_device_id, device_name FROM devices
),
daily_production_windows AS (
    SELECT
        dm.device_id,
        toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS production_date,
        argMax(dm.value, dm.timestamp) - argMin(dm.value, dm.timestamp) AS daily_production
    FROM device_metrics dm
    WHERE dm.parameter = 'length'
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') >= toDateTime('{start_date} 07:30:00', 'Asia/Colombo')
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') <= toDateTime('{end_date} 19:30:00', 'Asia/Colombo')
        {device_filter}
    GROUP BY dm.device_id, toDate(toTimeZone(dm.timestamp, 'Asia/Colombo'))
)
SELECT
    dl.device_name AS machine_name,
    dpw.production_date,
    dpw.daily_production
FROM daily_production_windows dpw
LEFT JOIN device_lookup dl ON dpw.device_id = dl.virtual_device_id
ORDER BY dpw.production_date, dl.device_name
""",

    "energy_consumption_hourly": """
WITH device_lookup AS (
    SELECT virtual_device_id, device_name FROM devices
),
hourly_consumption AS (
    SELECT
        dm.device_id,
        toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS date,
        toHour(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS hour,
        argMax(dm.value, dm.timestamp) - argMin(dm.value, dm.timestamp) AS hourly_consumption
    FROM device_metrics dm
    WHERE dm.parameter = 'TotalEnergy'
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') >= toDateTime('{start_date} 07:30:00', 'Asia/Colombo')
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') <= toDateTime('{end_date} 19:30:00', 'Asia/Colombo')
        {device_filter}
    GROUP BY dm.device_id, toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')), toHour(toTimeZone(dm.timestamp, 'Asia/Colombo'))
)
SELECT
    dl.device_name AS machine_name,
    hc.date,
    hc.hour,
    hc.hourly_consumption
FROM hourly_consumption hc
LEFT JOIN device_lookup dl ON hc.device_id = dl.virtual_device_id
ORDER BY hc.date, hc.hour, dl.device_name
""",

    "utilization_hourly": """
WITH device_lookup AS (
    SELECT virtual_device_id, device_name FROM devices
),
status_calculation AS (
    SELECT
        dm.device_id,
        toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS date,
        toHour(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS hour,
        sum(CASE WHEN dm.value = 0 THEN 1 ELSE 0 END) * 100.0 / count(*) AS efficiency_percent
    FROM device_metrics dm
    WHERE dm.parameter = 'status'
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') >= toDateTime('{start_date} 07:30:00', 'Asia/Colombo')
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') <= toDateTime('{end_date} 19:30:00', 'Asia/Colombo')
        {device_filter}
    GROUP BY dm.device_id, toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')), toHour(toTimeZone(dm.timestamp, 'Asia/Colombo'))
)
SELECT
    dl.device_name AS machine_name,
    sc.date,
    sc.hour,
    sc.efficiency_percent
FROM status_calculation sc
LEFT JOIN device_lookup dl ON sc.device_id = dl.virtual_device_id
ORDER BY sc.date, sc.hour, dl.device_name
"""
}

def detect_query_intent(user_query: str):
    """Detect which predefined query template to use"""
    user_query_lower = user_query.lower()
    
    # Production queries
    if any(word in user_query_lower for word in ['hourly production', 'production by hour', 'hour production']):
        return 'hourly_production', extract_date_range(user_query)
    
    elif any(word in user_query_lower for word in ['daily production', 'production by day', 'day production', 'production each day']):
        return 'daily_production', extract_date_range(user_query)
    
    # Energy queries
    elif any(word in user_query_lower for word in ['energy consumption', 'hourly energy', 'energy by hour']):
        return 'energy_consumption_hourly', extract_date_range(user_query)
    
    # Utilization queries
    elif any(word in user_query_lower for word in ['utilization', 'efficiency', 'uptime']):
        return 'utilization_hourly', extract_date_range(user_query)
    
    return None, None

def extract_date_range(user_query: str):
    """Extract date range from user query"""
    
    
    # Look for month names
    if 'april' in user_query.lower():
        return '2025-04-01', '2025-04-30'
    elif 'march' in user_query.lower():
        return '2025-03-01', '2025-03-31'
    elif 'may' in user_query.lower():
        return '2025-05-01', '2025-05-31'
    
    # Look for "last 7 days", "past week", etc.
    if any(phrase in user_query.lower() for phrase in ['last 7 days', 'past week', 'past 7 days']):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    # Default to current month
    now = datetime.now()
    start_date = now.replace(day=1).strftime('%Y-%m-%d')
    end_date = now.strftime('%Y-%m-%d')
    return start_date, end_date

def build_device_filter(user_query: str):
    """Build device filter based on user query"""
    user_query_lower = user_query.lower()
    
    if 'TJ-Stenter01' in user_query_lower:
        return "AND d.device_name = 'TJ-Stenter01'"
    elif 'TJ-Stenter02' in user_query_lower:
        return "AND d.device_name = 'TJ-Stenter02'"
    elif 'TJ-Stenter03' in user_query_lower:
        return "AND d.device_name = 'TJ-Stenter03'"
    elif any(phrase in user_query_lower for phrase in ['all machines', 'three machines', 'each machine']):
        return ""  # No filter for all machines
    
    return ""  # Default: no filter


# Configure logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()












# 2. REPLACE init_database function
def init_database(user: str, password: str, host: str, port: str, database: str):
    """Initialize ClickHouse connection with logging"""
    try:
        logger.info(f"Attempting to connect to ClickHouse: {host}:{port}/{database}")
        
        # Use clickhouse_connect for ClickHouse Cloud
        client = clickhouse_connect.get_client(
            host='q9gou7xoo6.ap-south-1.aws.clickhouse.cloud',
            user='shashini',
            password='78e76q}D7£6[',
            database='default'if database else None,
            secure=True  # Always use secure connection for ClickHouse Cloud
        )
        
        # Test connection
        result = client.query('SELECT 1')
        logger.info("ClickHouse connection successful")
        return client
    except Exception as e:
        logger.error(f"ClickHouse connection failed: {str(e)}")
        raise e














    
def detect_visualization_request(user_query: str):
    """Enhanced visualization detection with better single and multi-machine support"""
    user_query_lower = user_query.lower()
    logger.info(f"Analyzing query for visualization: {user_query}")
    

    # Check if using a template
    query_template, _ = detect_query_intent(user_query)


    # Keywords that indicate visualization request
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot',
        'bar', 'line', 'pie', 'trend', 'comparison', 'grouped bar', 'stacked bar',
        'pulse', 'pulse per minute', 'pulse rate', 'pulse variation', 'pulse trend'
    ]
    
    needs_viz = any(keyword in user_query_lower for keyword in viz_keywords)
    # Template-specific chart types
    if query_template == 'daily_production':
        chart_type = "multi_machine_bar" if 'all machines' in user_query_lower else "bar"
    elif query_template == 'hourly_production':
        chart_type = "line"
    elif query_template == 'utilization_hourly':
        chart_type = "line"
    else:
        # Existing chart type detection logic
        chart_type = "bar"  # default

    
    
    
    # Check for multi-machine/multi-category requests
    multi_machine_keywords = ['all machines', 'each machine', 'by machine', 'machines production', 'three machines']
    multi_category_keywords = ['by day', 'daily', 'monthly', 'by month','each day', 'each month']
    
    
    is_multi_machine = any(keyword in user_query_lower for keyword in multi_machine_keywords)
    is_multi_category = any(keyword in user_query_lower for keyword in multi_category_keywords)
    
    if is_multi_machine and (is_multi_category or 'bar' in user_query_lower):
        chart_type = "multi_machine_bar"
    elif any(word in user_query_lower for word in ['line', 'trend', 'over time', 'hourly', 'daily', 'time series']):
        chart_type = "line"
    elif any(word in user_query_lower for word in ['pie', 'proportion', 'percentage', 'share', 'distribution']):
        chart_type = "pie"
    elif any(word in user_query_lower for word in ['scatter', 'relationship', 'correlation']):
        chart_type = "scatter"
    elif any(word in user_query_lower for word in ['histogram', 'distribution', 'frequency']):
        chart_type = "histogram"
    elif any(word in user_query_lower for word in ['grouped', 'stacked', 'multiple']):
        chart_type = "grouped_bar"
    elif any(word in user_query_lower for word in ['pulse', 'pulse per minute', 'rate per minute']):
        chart_type = "pulse_line"  # New chart type for pulse data
        
    logger.info(f"Visualization needed: {needs_viz}, Chart type: {chart_type}, Multi-machine: {is_multi_machine}")
    return needs_viz, chart_type








# Enhanced SQL Chain with better instructions based on your PDF queries

def get_enhanced_sql_chain(db):
    """Enhanced SQL chain with real query examples from queries.pdf"""
    template = """
    You are an expert ClickHouse analyst for production monitoring systems. 
    Generate ClickHouse queries that follow the EXACT patterns from the production system queries.

    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}

    CRITICAL CLICKHOUSE PRODUCTION QUERY PATTERNS:
    
    1. ALWAYS use these exact CTEs and structure patterns:
    
    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    production_calculation_query AS (
        WITH hourly_windows AS (
            SELECT
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                device_id,
                value,
                -- Time bucket logic here
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('YYYY-MM-DD 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('YYYY-MM-DD 19:30:00', 'Asia/Colombo')
        )
        -- Additional CTEs here
    )
    SELECT 
        dl.device_name AS machine_name,
        -- other columns
    FROM production_calculation_query pcq
    LEFT JOIN device_lookup dl ON pcq.device_id = dl.virtual_device_id
    ORDER BY date, machine_name
    ```

    2. TIME BUCKET PATTERNS (Copy exactly from PDF):
    For hourly queries, use EXACT time buckets:
    ```sql
    CASE
        WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 7 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '07:30-08:30'
        WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 8 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '07:30-08:30'
        -- Continue with all 12 hourly slots exactly as in PDF
        ELSE NULL
    END AS hour_bucket
    ```

    3. PRODUCTION CALCULATION PATTERNS:
    ```sql
    argMax(value, timestamp) - argMin(value, timestamp) AS production_output
    ```
    
    4. ENERGY CONSUMPTION PATTERNS:
    Use parameter = 'TotalEnergy' (not 'energy')
    
    5. UTILIZATION PATTERNS:
    Use parameter = 'status' with time gap calculations:
    ```sql
    sum(CASE WHEN value = 0 THEN interval_duration ELSE 0 END) AS on_time_seconds
    ```

    # Add this section to your template string, right after the existing examples:

    4. PROPER LEAD/LAG WINDOW FUNCTION PATTERNS:
    CRITICAL: Never use NULL as default value in LEAD/LAG functions. Always provide proper typed defaults:
    
    ```sql
    -- CORRECT: Use proper datetime default
    lead(sl_timestamp, 1, toDateTime('2025-04-30 19:30:00', 'Asia/Colombo')) OVER (PARTITION BY device_id ORDER BY sl_timestamp) AS next_timestamp
    
    -- WRONG: Using NULL
    lead(sl_timestamp, 1, NULL) OVER (...)
    ```
    
    5. UTILIZATION CALCULATION PATTERNS:
    For status-based utilization, use this exact pattern:
    ```sql
    WITH status_with_next AS (
        SELECT
            device_id,
            toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
            value,
            toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
            lead(toTimeZone(timestamp, 'Asia/Colombo'), 1, 
                 toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 19:30:00'), 'Asia/Colombo')
            ) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS next_sl_timestamp
        FROM device_metrics
        WHERE parameter = 'status'
    )
    ```


    REAL PRODUCTION EXAMPLES FROM YOUR SYSTEM:

    Example 1 - Hourly Production:
    Question: "Show hourly production for all machines in April"
    Answer:
    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    production_calculation_query AS (
        WITH hourly_windows AS (
            SELECT
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                device_id,
                value,
                CASE
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 7 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '07:30-08:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 8 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '07:30-08:30'
                    -- Add all other time slots
                    ELSE NULL
                END AS hour_bucket,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-04-01 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-04-30 19:30:00', 'Asia/Colombo')
        ),
        production_windows AS (
            SELECT
                device_id,
                hour_bucket,
                date,
                argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS production_output
            FROM hourly_windows
            WHERE hour_bucket IS NOT NULL
            GROUP BY device_id, hour_bucket, date
        )
    )
    SELECT 
        dl.device_name AS machine_name,
        pcq.hour_bucket AS time_slot,
        pcq.date AS production_date,
        pcq.production_output AS hourly_production
    FROM production_calculation_query pcq
    LEFT JOIN device_lookup dl ON pcq.device_id = dl.virtual_device_id
    ORDER BY pcq.date, pcq.hour_bucket, dl.device_name
    ```

    Example 2 - Daily Production:
    Question: "Show daily production by machine"
    Answer:
    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    daily_production_windows AS (
        SELECT
            dm.device_id,
            toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS production_date,
            argMax(dm.value, dm.timestamp) - argMin(dm.value, dm.timestamp) AS daily_production
        FROM device_metrics dm
        WHERE dm.parameter = 'length'
            AND toTimeZone(dm.timestamp, 'Asia/Colombo') >= toDateTime('2025-04-01 07:30:00', 'Asia/Colombo')
            AND toTimeZone(dm.timestamp, 'Asia/Colombo') <= toDateTime('2025-04-30 19:30:00', 'Asia/Colombo')
        GROUP BY dm.device_id, toDate(toTimeZone(dm.timestamp, 'Asia/Colombo'))
    )
    SELECT
        dl.device_name AS machine_name,
        dpw.production_date,
        dpw.daily_production
    FROM daily_production_windows dpw
    LEFT JOIN device_lookup dl ON dpw.device_id = dl.virtual_device_id
    ORDER BY dpw.production_date, dl.device_name
    ```

    Example 3 - Energy Consumption:
    Question: "Show energy consumption by hour"
    Answer:
    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    hourly_consumption AS (
        SELECT
            dm.device_id,
            toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS date,
            toHour(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS hour,
            argMax(dm.value, dm.timestamp) - argMin(dm.value, dm.timestamp) AS hourly_consumption
        FROM device_metrics dm
        WHERE dm.parameter = 'TotalEnergy'
            AND toTimeZone(dm.timestamp, 'Asia/Colombo') >= toDateTime('2025-04-01 07:30:00', 'Asia/Colombo')
            AND toTimeZone(dm.timestamp, 'Asia/Colombo') <= toDateTime('2025-04-30 19:30:00', 'Asia/Colombo')
        GROUP BY dm.device_id, toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')), toHour(toTimeZone(dm.timestamp, 'Asia/Colombo'))
    )
    SELECT
        dl.device_name AS machine_name,
        hc.date,
        hc.hour,
        hc.hourly_consumption
    FROM hourly_consumption hc
    LEFT JOIN device_lookup dl ON hc.device_id = dl.virtual_device_id
    ORDER BY hc.date, hc.hour, dl.device_name
    ```

    MANDATORY REQUIREMENTS:
    1. ALWAYS use 'Asia/Colombo' timezone
    2. ALWAYS filter between 07:30:00 and 19:30:00 for working hours
    3. ALWAYS use proper parameter names: 'length' for production, 'TotalEnergy' for energy, 'status' for utilization
    4. ALWAYS use device_lookup CTE pattern
    5. ALWAYS use argMax/argMin for counter calculations
    6. ALWAYS use proper column aliases: machine_name, production_date, daily_production, etc.
    7. NEVER use simple SUM() for counters - always use argMax - argMin pattern
    8. CRITICAL: NEVER use NULL in LEAD/LAG functions - always provide typed defaults:
       - For datetime: toDateTime('YYYY-MM-DD HH:MM:SS', 'Asia/Colombo')
       - For dates: use end of period date
       - For numbers: use 0 or appropriate default value
    9. For multi-day utilization queries, use proper date-based partitioning in window functions
    10. Always handle edge cases in time calculations with greatest(0, calculation) to avoid negative values


    


    Write ONLY the ClickHouse SQL query. No explanations, no backticks, no markdown formatting.


    
    
    Question: {question}
    SQL Query:
    """








    ENHANCED_METABASE_TEMPLATES = {
        # Copy your existing templates and add these enhancements
        
        "hourly_production_full": """
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    production_calculation_query AS (
        WITH hourly_windows AS (
            SELECT
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                device_id,
                value,
                CASE
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 7 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '07:30-08:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 8 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '07:30-08:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 8 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '08:30-09:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 9 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '08:30-09:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 9 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '09:30-10:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 10 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '09:30-10:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 10 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '10:30-11:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 11 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '10:30-11:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 11 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '11:30-12:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 12 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '11:30-12:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 12 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '12:30-13:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 13 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '12:30-13:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 13 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '13:30-14:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 14 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '13:30-14:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 14 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '14:30-15:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 15 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '14:30-15:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 15 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '15:30-16:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 16 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '15:30-16:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 16 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '16:30-17:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 17 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '16:30-17:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 17 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '17:30-18:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 18 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '17:30-18:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 18 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '18:30-19:30'
                    WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 19 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '18:30-19:30'
                    ELSE NULL
                END AS hour_bucket,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('{start_date} 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('{end_date} 19:30:00', 'Asia/Colombo')
                {device_filter}
        ),
        production_windows AS (
            SELECT
                device_id,
                hour_bucket,
                date,
                argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS production_output
            FROM hourly_windows
            WHERE hour_bucket IS NOT NULL
            GROUP BY device_id, hour_bucket, date
        )
    )
    SELECT 
        dl.device_name AS machine_name,
        pcq.hour_bucket AS time_slot,
        pcq.date AS production_date,
        pcq.production_output AS hourly_production
    FROM production_calculation_query pcq
    LEFT JOIN device_lookup dl ON pcq.device_id = dl.virtual_device_id
    ORDER BY pcq.date, pcq.hour_bucket, dl.device_name
    """,

        "utilization_hourly_full": """
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    status_calculation_query AS (
        WITH time_gaps AS (
            SELECT
                timestamp,
                device_id,
                parameter,
                value,
                dateDiff('second', lagInFrame(timestamp, 1) OVER (PARTITION BY device_id ORDER BY timestamp), timestamp) AS time_gap_seconds,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp
            FROM device_metrics
            WHERE parameter = 'status'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('{start_date} 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('{end_date} 19:30:00', 'Asia/Colombo')
                {device_filter}
            ORDER BY device_id, timestamp
        ),
        status_changes AS (
            SELECT
                device_id,
                value AS current_status,
                time_gap_seconds AS interval_duration,
                CASE
                    WHEN toHour(sl_timestamp) = 7 AND toMinute(sl_timestamp) >= 30 THEN '07:30-08:30'
                    WHEN toHour(sl_timestamp) = 8 AND toMinute(sl_timestamp) < 30 THEN '07:30-08:30'
                    -- Add all time buckets like in production query
                    ELSE NULL
                END AS hour_bucket,
                toDate(sl_timestamp) AS date
            FROM time_gaps
        ),
        hourly_aggregation AS (
            SELECT
                device_id,
                hour_bucket,
                date,
                sum(CASE WHEN current_status = 0 THEN interval_duration ELSE 0 END) AS on_time_seconds,
                3600 AS total_window_time_seconds
            FROM status_changes
            WHERE hour_bucket IS NOT NULL
            GROUP BY device_id, hour_bucket, date
        )
    )
    SELECT
        dl.device_name AS machine_name,
        scq.date,
        scq.hour_bucket,
        CASE
            WHEN scq.total_window_time_seconds > 0 THEN round((scq.on_time_seconds / scq.total_window_time_seconds) * 100, 2)
            ELSE 0
        END AS efficiency_percent
    FROM status_calculation_query scq
    LEFT JOIN device_lookup dl ON scq.device_id = dl.virtual_device_id
    ORDER BY scq.date, scq.hour_bucket, dl.device_name
    """
    }



    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import AzureChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    import os
        
    prompt = ChatPromptTemplate.from_template(template)
        
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0,
        max_tokens=30000,  # Increased for complex queries
        )
        
    def get_schema(_):
            return """
            Tables:
            - devices: virtual_device_id (Int64), device_name (String)
            - device_metrics: device_id (Int64), timestamp (DateTime), parameter (String), value (Float64)
            
            Parameters:
            - 'length': Production counter values (use argMax-argMin for production calculation)
            - 'TotalEnergy': Energy consumption counter values
            - 'status': Machine status (0=ON, 1=OFF) for utilization calculations
                
            Time Zones:
            - All queries use 'Asia/Colombo' timezone
            - Working hours: 07:30:00 to 19:30:00
            """
        
    return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
        )    



    
    
    
    
    
    
    
    

def create_enhanced_visualization(df, chart_type, user_query):
    """Enhanced visualization with proper single and multi-machine support"""
    try:
        logger.info(f"Creating visualization: {chart_type}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame sample data:\n{df.head()}")
        
        if df.empty:
            logger.warning("DataFrame is empty")
            st.warning("No data available for visualization")
            return False
        
        # Clean and prepare data
        df = df.dropna()
        if df.empty:
            logger.warning("DataFrame is empty after removing NaN values")
            st.warning("No valid data available after cleaning")
            return False
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle datetime columns
        for col in df.columns:
            if df[col].dtype == 'object' and col.lower() in ['production_date', 'day', 'date', 'production_month', 'month']:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Converted column {col} to datetime")
                except:
                    pass
        
        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        fig = None
        
        
        
        # Enhanced multi-machine bar chart
        if chart_type == "multi_machine_bar" or chart_type == "grouped_bar":
            # Look for standard column patterns
            date_col = None
            machine_col = None
            value_col = None
            
            # Find date column
            for col in df.columns:
                if col.lower() in ['production_date', 'date', 'day', 'production_month', 'month', 'hour']:
                    date_col = col
                    break
            
            # Find machine column
            for col in df.columns:
                if col.lower() in ['machine_name', 'device_name', 'machine', 'device']:
                    machine_col = col
                    break
            
            # Find value column
            for col in df.columns:
                if col.lower() in ['daily_production', 'monthly_production', 'production_output', 'total_output', 'production']:
                    value_col = col
                    break
                elif col in numeric_cols:
                    value_col = col
                    break
            
            logger.info(f"Detected columns - Date: {date_col}, Machine: {machine_col}, Value: {value_col}")
            
            if date_col and machine_col and value_col:
                # Create grouped bar chart
                fig = px.bar(
                    df, 
                    x=date_col, 
                    y=value_col, 
                    color=machine_col,
                    title=f"Production by Machine and Date",
                    labels={
                        date_col: "Date",
                        value_col: "Production Output",
                        machine_col: "Machine"
                    },
                    barmode='group'
                )
                
                # Enhance the chart appearance
                fig.update_layout(
                    showlegend=True,
                    height=600,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=16,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Improve x-axis for dates
                if df[date_col].dtype == 'datetime64[ns]':
                    fig.update_xaxes(
                        tickangle=45,
                        tickformat='%Y-%m-%d'
                    )
                
            else:
                # Fallback to regular grouped bar chart
                if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                    x_col = categorical_cols[0]
                    color_col = categorical_cols[1]
                    y_col = numeric_cols[0]
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                               title=f"Grouped Bar Chart: {y_col} by {x_col} and {color_col}",
                               barmode='group')
                    
                    
                    
                    
                    
                    
        
        # Replace the line chart section in your create_enhanced_visualization function

        elif chart_type == "line":
            # Enhanced logic for line charts with proper date handling
            date_col = None
            value_col = None
            color_col = None
            
            # Find date column first (this should be X-axis)
            for col in df.columns:
                if col.lower() in ['production_date', 'date', 'day', 'production_month', 'month', 'timestamp', 'time']:
                    date_col = col
                    break
            
            # Find machine column for color grouping
            for col in df.columns:
                if col.lower() in ['machine_name', 'device_name', 'machine', 'device']:
                    color_col = col
                    break
            
            # Find numeric value column for Y-axis
            for col in df.columns:
                if col.lower() in ['daily_production', 'monthly_production', 'production_output', 'total_output', 'production', 'pulse_per_minute', 'pulse']:
                    value_col = col
                    break
                elif col in numeric_cols and col != date_col:  # Any numeric column except date
                    value_col = col
                    break
            
            # Fallback logic if specific columns not found
            if not date_col and len(df.columns) >= 2:
                # Look for any datetime column or the first non-machine column
                for col in df.columns:
                    if df[col].dtype == 'datetime64[ns]' or col.lower() not in ['machine_name', 'device_name']:
                        date_col = col
                        break
                if not date_col:
                    date_col = df.columns[0]  # Last resort
            
            if not value_col:
                value_col = numeric_cols[0] if numeric_cols else df.columns[-1]
            
            logger.info(f"Line chart - Date: {date_col}, Value: {value_col}, Color: {color_col}")
            
            # Convert date column to datetime if it's not already
            if date_col and df[date_col].dtype != 'datetime64[ns]':
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    logger.info(f"Converted {date_col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {date_col} to datetime: {e}")
            
            # Create the line chart
            fig = px.line(
                df, 
                x=date_col, 
                y=value_col, 
                color=color_col,
                title=f"Line Chart: {value_col} over {date_col}",
                labels={
                    date_col: "Date",
                    value_col: value_col.replace('_', ' ').title(),
                    color_col: "Machine" if color_col else None
                }
            )
            
            # Enhanced formatting for line charts
            fig.update_layout(
                height=600,
                showlegend=True if color_col else False,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                title_font_size=16
            )
            
            # Format x-axis for dates
            if df[date_col].dtype == 'datetime64[ns]':
                fig.update_xaxes(
                    tickangle=45,
                    tickformat='%Y-%m-%d'
                )
            
            # Add markers to make lines more visible
            fig.update_traces(mode='lines+markers', marker_size=6)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                labels_col = categorical_cols[0] if categorical_cols else df.columns[0]
                values_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.pie(df, names=labels_col, values=values_col, 
                           title=f"Pie Chart: {values_col} by {labels_col}")
                
        elif chart_type == "bar":
            if len(df.columns) >= 2:
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.bar(df, x=x_col, y=y_col, 
                           title=f"Bar Chart: {y_col} by {x_col}")
        
        elif chart_type == "pulse_line":
            # Look for pulse-specific columns
            time_col = None
            pulse_col = None
            machine_col = None
    
            # Find timestamp column
            for col in df.columns:
                if col.lower() in ['timestamp', 'time', 'datetime']:
                    time_col = col
                    break
    
            # Find pulse column
            for col in df.columns:
                if col.lower() in ['pulse_per_minute', 'pulse', 'pulse_rate']:
                    pulse_col = col
                    break
    
            # Find machine column
            for col in df.columns:
                if col.lower() in ['device_name', 'machine_name']:
                    machine_col = col
                    break
    
            if time_col and pulse_col:
                fig = px.line(
                    df, 
                    x=time_col, 
                    y=pulse_col, 
                    color=machine_col,
                    title="Pulse Per Minute Over Time",
                    labels={
                        time_col: "Time",
                        pulse_col: "Pulse Per Minute",
                        machine_col: "Machine"
                    }
                )
        
                # Enhanced formatting for pulse charts
                fig.update_layout(
                    height=600,
                    showlegend=True,
                    yaxis_title="Pulse Per Minute"
                )
        
                fig.update_xaxes(
                    tickangle=45,
                    tickformat='%H:%M'
                )
        
        # Display the chart
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            logger.info("Visualization created successfully")
            
            # Show data summary
            with st.expander("📊 Data Summary"):
                st.write(f"**Total records:** {len(df)}")
                if 'machine_name' in df.columns or 'device_name' in df.columns:
                    machine_col = 'machine_name' if 'machine_name' in df.columns else 'device_name'
                    unique_machines = df[machine_col].nunique()
                    st.write(f"**Unique machines:** {unique_machines}")
                    st.write(f"**Machines:** {', '.join(df[machine_col].unique())}")
                
                # Show efficiency statistics if it's an efficiency chart
                if chart_type == "efficiency_bar" and 'efficiency_percent' in df.columns:
                    st.write(f"**Average efficiency:** {df['efficiency_percent'].mean():.1f}%")
                    st.write(f"**Highest efficiency:** {df['efficiency_percent'].max():.1f}%")
                    st.write(f"**Lowest efficiency:** {df['efficiency_percent'].min():.1f}%")
                
                st.dataframe(df)
            
            return True
        else:
            error_msg = f"Could not create {chart_type} chart with available data."
            logger.error(error_msg)
            st.error(error_msg)
            st.write("**Available data:**")
            st.dataframe(df.head(10))
            return False
            
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.write("**Debug information:**")
        st.write("Data sample:")
        st.dataframe(df.head() if not df.empty else pd.DataFrame())
        return False

def is_greeting_or_casual(user_query: str) -> bool:
    """Detect if the user query is a greeting or casual conversation"""
    user_query_lower = user_query.lower().strip()
    
    # Common greetings and casual phrases
    greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'whats up', "what's up", 'yo', 'hiya', 'greetings'
    ]
    
    casual_phrases = [
        'thank you', 'thanks', 'bye', 'goodbye', 'see you', 'ok', 'okay',
        'cool', 'nice', 'great', 'awesome', 'perfect', 'got it', 'understand',
        'help', 'what can you do', 'how does this work', 'test'
    ]
    
    # Check if the query is just a greeting or casual phrase
    if user_query_lower in greetings + casual_phrases:
        return True
    
    # Check if query starts with greeting
    if any(user_query_lower.startswith(greeting) for greeting in greetings):
        return True
    
    # Check if it's a very short query without data-related keywords
    data_keywords = [
        'production', 'machine', 'data', 'show', 'chart', 'graph', 'plot',
        'select', 'table', 'database', 'query', 'april', 'month', 'day',
        'output', 'performance', 'efficiency', 'downtime', 'shift','pulse', 'pulse per minute', 'rate', 'length', 'variation', 'trend'
    ]
    
    if len(user_query_lower.split()) <= 3 and not any(keyword in user_query_lower for keyword in data_keywords):
        return True
    
    return False

def get_casual_response(user_query: str) -> str:
    """Generate appropriate responses for greetings and casual conversation"""
    user_query_lower = user_query.lower().strip()
    
    if any(greeting in user_query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return """Hello! 👋 I'm your Production Analytics Bot. I'm here to help you analyze your production data and create visualizations."""

    elif any(phrase in user_query_lower for phrase in ['how are you', 'whats up', "what's up"]):
        return """I'm doing great, thank you! 😊 Ready to help you dive into your production data.

Is there any specific production analysis or visualization you'd like me to help you with?"""

    elif any(phrase in user_query_lower for phrase in ['thank you', 'thanks']):
        return """You're welcome! 😊 I'm here whenever you need help with production data analysis or creating visualizations.

Feel free to ask me about any production metrics you'd like to explore!"""

    elif any(phrase in user_query_lower for phrase in ['help', 'what can you do', 'how does this work']):
        return """I'm your Production Analytics Assistant! Here's how I can help:


Just ask me a question about your production data, and I'll generate both the analysis and visualizations for you!"""

    elif user_query_lower in ['test', 'testing']:
        return """System test successful! ✅ 


What would you like to analyze?"""

    else:
        return """I'm here to help with production data analysis and visualizations! 


What production data would you like to explore? 📊"""













# ADD THESE NEW FUNCTIONS:

def generate_session_title(first_message: str) -> str:
    """Generate a meaningful session title from the first user message"""
    message_lower = first_message.lower().strip()
    
    # Extract key topics
    if 'production' in message_lower:
        if 'april' in message_lower:
            return "April Production Analysis"
        elif 'may' in message_lower:
            return "May Production Analysis"
        elif 'march' in message_lower:
            return "March Production Analysis"
        elif 'daily' in message_lower:
            return "Daily Production Report"
        elif 'hourly' in message_lower:
            return "Hourly Production Analysis"
        else:
            return "Production Analysis"
    
    elif 'energy' in message_lower:
        return "Energy Consumption Analysis"
    
    elif 'utilization' in message_lower or 'efficiency' in message_lower:
        return "Machine Utilization Report"
    
    elif 'pulse' in message_lower:
        return "Pulse Rate Analysis"
    
    elif any(word in message_lower for word in ['chart', 'graph', 'plot', 'visualize']):
        return "Data Visualization Request"
    
    # Fallback: use first few words
    words = first_message.split()[:4]
    if len(words) > 0:
        return " ".join(words).title()
    
    return f"Chat Session"

def trim_chat_history(messages: list, max_messages: int = 10) -> list:
    """Keep only the last N messages"""
    if len(messages) <= max_messages:
        return messages
    
    # Always keep the first message (initial greeting) and last 9 messages
    if len(messages) > max_messages:
        return [messages[0]] + messages[-(max_messages-1):]
    return messages

def initialize_session_state():
    """Initialize session state with proper structure"""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    
    if "active_session_id" not in st.session_state:
        # Create initial session
        session_id = str(uuid.uuid4())
        st.session_state.sessions[session_id] = {
            "title": "Welcome Chat",
            "messages": [
                AIMessage(content="Hello! I'm your Althinect Intelligence Bot. I can help you analyze multi-machine production data with colorful visualizations! 📊\n\nTry asking: *'Show all three machines production by each machine in April with bar chart for all 30 day'*")
            ],
            "created_at": datetime.now()
        }
        st.session_state.active_session_id = session_id

def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "title": "New Chat",
        "messages": [
            AIMessage(content="Hello! I'm your Althinect Intelligence Bot. I can help you analyze multi-machine production data with colorful visualizations! 📊")
        ],
        "created_at": datetime.now()
    }
    st.session_state.active_session_id = session_id
    st.rerun()

def get_current_session_messages():
    """Get messages from the current active session"""
    if st.session_state.active_session_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.active_session_id]["messages"]
    return []

def add_message_to_current_session(message):
    """Add a message to the current session and trim if necessary"""
    if st.session_state.active_session_id in st.session_state.sessions:
        current_messages = st.session_state.sessions[st.session_state.active_session_id]["messages"]
        current_messages.append(message)
        
        # Trim to last 10 messages
        st.session_state.sessions[st.session_state.active_session_id]["messages"] = trim_chat_history(current_messages, 10)

def update_session_title(session_id: str, first_user_message: str):
    """Update session title based on first user message"""
    if session_id in st.session_state.sessions:
        # Only update if it's still the default title
        if st.session_state.sessions[session_id]["title"] in ["New Chat", "Welcome Chat"]:
            new_title = generate_session_title(first_user_message)
            st.session_state.sessions[session_id]["title"] = new_title


























def get_enhanced_response(user_query: str, db, chat_history: list):
    try:
        logger.info(f"Processing user query: {user_query}")
        
        # Step 0: Check if this is a greeting or casual conversation
        if is_greeting_or_casual(user_query):
            logger.info("Detected greeting/casual conversation")
            return get_casual_response(user_query)

        

        # Step 1.5: Check for predefined query templates
        query_template, date_range = detect_query_intent(user_query)
        if query_template and query_template in METABASE_QUERY_TEMPLATES:
            logger.info(f"Using predefined query template: {query_template}")
            
            start_date, end_date = date_range
            device_filter = build_device_filter(user_query)
            
            # Format the template with parameters
            sql_query = METABASE_QUERY_TEMPLATES[query_template].format(
                start_date=start_date,
                end_date=end_date,
                device_filter=device_filter
            )
            
            logger.info(f"Using template query: {query_template}")
        else:
            # Existing logic: Generate SQL query with enhanced chain
            sql_chain = get_enhanced_sql_chain(db)
            sql_query = sql_chain.invoke({
                "question": user_query,
                "chat_history": chat_history
            })





        
        # Step 1: Detect visualization needs
        needs_viz, chart_type = detect_visualization_request(user_query)
        
        # Step 2: Generate SQL query with enhanced chain
        sql_chain = get_enhanced_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        
        logger.info(f"Generated ClickHouse SQL query: {sql_query}")
        
        # Step 3: Execute ClickHouse query with correct method
        try:
            # FIXED: Use .query() method instead of .execute() for clickhouse_connect
            result = db.query(sql_query)
            
            # Extract the data from the result
            if hasattr(result, 'result_set'):
                sql_response = result.result_set
                column_names = result.column_names if hasattr(result, 'column_names') else []
            else:
                # Some versions return data directly
                sql_response = result
                column_names = []
            
            logger.info(f"ClickHouse query executed successfully. Response length: {len(sql_response) if sql_response else 0}")
            
            # Check if response is empty
            if not sql_response or len(sql_response) == 0:
                return "No data found for your query. This could be due to:\n\n1. **Date range issue**: The specified date range might not have data\n2. **Table structure**: Column names might be different\n3. **Data availability**: No records match your criteria\n\nPlease try:\n- Checking if data exists for the specified time period\n- Using a different date range\n- Asking about available tables or columns"
            
        except Exception as e:
            error_msg = f"ClickHouse execution error: {str(e)}\n\n**Generated ClickHouse Query:**\n```sql\n{sql_query}\n```\n\n**Possible issues:**\n1. Column names might be incorrect\n2. Table structure might be different\n3. Date format issues\n4. ClickHouse function usage errors"
            logger.error(error_msg)
            return error_msg
        
        # Step 4: Create visualization if needed
        chart_created = False
        if needs_viz:
            try:
                # Convert ClickHouse result to DataFrame
                if sql_response:
                    # Create DataFrame with proper column names
                    if column_names:
                        df = pd.DataFrame(sql_response, columns=column_names)
                    else:
                        # Fallback: create DataFrame and try to infer column names
                        df = pd.DataFrame(sql_response)
                        
                        # Try to infer column names from SQL query
                        if "AS machine_name" in sql_query and "AS production_date" in sql_query:
                            expected_cols = ['machine_name', 'production_date', 'daily_production']
                            if len(df.columns) == len(expected_cols):
                                df.columns = expected_cols
                        
                    logger.info(f"DataFrame created with shape: {df.shape}")
                    logger.info(f"DataFrame columns: {list(df.columns)}")
                    
                    if df.empty:
                        st.warning("Query returned no data for visualization")
                    else:
                        chart_created = create_enhanced_visualization(df, chart_type, user_query)
                        
            except Exception as e:
                error_msg = f"Visualization error: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        
        # Step 5: Generate natural language response
        template = """
        You are a data analyst providing insights about production data.
        
        Based on the SQL query results, provide a clear, informative response.
    
        SQL Query: {query}
        User Question: {question}
        SQL Response: {response}
        
        {visualization_note}
        
        Guidelines:
        1. Summarize key findings from the data
        2. Mention specific numbers/values when relevant
        3. If this is multi-machine data, highlight comparisons between machines
        4. If this is time-series data, mention trends or patterns
        5. Keep the response concise but informative
        6. If this is pulse data, explain the pulse calculation shows machine activity rate
        """
        
        visualization_note = ""
        if needs_viz and chart_created:
            visualization_note = "Note: The visualization above shows the data in an interactive chart format with different colors for each machine."
        elif needs_viz and not chart_created:
            visualization_note = "Note: I attempted to create a visualization but encountered formatting issues. The raw data is available above."
        
        prompt = ChatPromptTemplate.from_template(template)
    
        # Fixed Azure OpenAI configuration
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
            max_tokens=800,
        )
        
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "question": user_query,
            "query": sql_query,
            "response": sql_response,
            "visualization_note": visualization_note
        })
        
        logger.info("Response generated successfully")
        return response
        
    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Streamlit UI


load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
    st.error("⚠️ AZURE_OPENAI_API_KEY key not found. Please add AZURE_OPENAI_API_KEY to your .env file.")
    st.stop()

st.set_page_config(page_title="Althinect Intelligence Bot", page_icon="📊")

st.title("Althinect Intelligence Bot")



# ✅ Auto-connect to ClickHouse at startup
if "db" not in st.session_state:
    try:
        db = init_database(
            user="shashini",
            password="78e76q}D7£6[",
            host="q9gou7xoo6.ap-south-1.aws.clickhouse.cloud",
            port="9440",
            database="default"
        )
        st.session_state.db = db
        logger.info("✅ ClickHouse connected automatically at startup")
    except Exception as e:
        st.error(f"❌ Auto connection failed: {str(e)}")
        logger.error(f"ClickHouse auto connection failed: {str(e)}")







# Initialize session state
initialize_session_state()

# Sidebar for session management
with st.sidebar:
    if "db" in st.session_state:
        st.success("🟢 Database Connected")
    else:
        st.warning("🔴 Database Not Connected")

        
    st.header("💬 Chat Sessions")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session()
    
    st.divider()
    
    # Display all sessions
    sessions_sorted = sorted(
        st.session_state.sessions.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )
    
    for session_id, session_data in sessions_sorted:
        # Create a container for each session
        is_active = session_id == st.session_state.active_session_id
        
        # Session button with title
        if st.button(
            f"{'🟢' if is_active else '⚪'} {session_data['title'][:25]}{'...' if len(session_data['title']) > 25 else ''}",
            key=f"session_{session_id}",
            use_container_width=True,
            disabled=is_active
        ):
            st.session_state.active_session_id = session_id
            st.rerun()
        
        # Show message count and last activity for active session
        if is_active:
            message_count = len(session_data['messages'])
            st.caption(f"📝 {message_count} messages")
    



    

    st.divider()
    
    # Session info
    if st.session_state.active_session_id in st.session_state.sessions:
        current_session = st.session_state.sessions[st.session_state.active_session_id]
        st.write("**Current Session:**")
        st.write(f"📋 {current_session['title']}")
        st.write(f"💬 {len(current_session['messages'])} messages")
        st.write(f"🕒 Created: {current_session['created_at'].strftime('%H:%M')}")

# Get current session messages
current_messages = get_current_session_messages()

# Display current session messages
for message in current_messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)

# Chat input
user_query = st.chat_input("💬 Ask about multi-machine production data...")
if user_query is not None and user_query.strip() != "":
    logger.info(f"User query received: {user_query}")
    
    # Check if this is the first user message in the session
    current_messages = get_current_session_messages()
    user_message_count = sum(1 for msg in current_messages if isinstance(msg, HumanMessage))
    
    # If this is the first user message, update the session title
    if user_message_count == 0:
        update_session_title(st.session_state.active_session_id, user_query)
    
    # Add user message to current session
    add_message_to_current_session(HumanMessage(content=user_query))
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    with st.chat_message("assistant", avatar="🤖"):
        # Check if it's a greeting first (no database needed)
        if is_greeting_or_casual(user_query):
            response = get_casual_response(user_query)
            st.markdown(response)
        elif "db" in st.session_state:
            with st.spinner("🔄 Analyzing data and creating visualization..."):
                # Get the last 10 messages from current session for context
                context_messages = get_current_session_messages()
                context_messages = trim_chat_history(context_messages, 10)
                
                response = get_enhanced_response(user_query, st.session_state.db, context_messages)
                st.markdown(response)
        else:
            response = "⚠️ Please connect to the database first using the sidebar to analyze production data."
            st.markdown(response)
            logger.warning("User attempted to query without database connection")
    
    # Add bot response to current session
    add_message_to_current_session(AIMessage(content=response))
    
    logger.info("Conversation turn completed")

# Optional: Add session management in sidebar
with st.sidebar:
    st.divider()
    if st.button("🗑️ Clear Current Session", use_container_width=True):
        if st.session_state.active_session_id in st.session_state.sessions:
            # Reset current session messages but keep the session
            st.session_state.sessions[st.session_state.active_session_id]["messages"] = [
                AIMessage(content="Hello! I'm your Althinect Intelligence Bot. How can I help you analyze your production data?")
            ]
            st.rerun()