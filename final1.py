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

}




# Machine mapping definitions
MACHINE_MAPPINGS = {
    'production': {
        'stenter1': 'TJ-Stenter01 Length(ioid2)',
        'stenter01': 'TJ-Stenter01 Length(ioid2)',  # Handle both formats
        'stenter2': 'TJ-Stenter02 Length(ioid1)', 
        'stenter02': 'TJ-Stenter02 Length(ioid1)',
        'stenter3': 'TJ-Stenter03 Length(ioid1)',
        'stenter03': 'TJ-Stenter03 Length(ioid1)',
        'stenter4': 'TJ-Stenter04 Length',
        'stenter04': 'TJ-Stenter04 Length',
        'stenter5': 'TJ-Stenter05 Length',
        'stenter05': 'TJ-Stenter05 Length',
        'stenter6': 'TJ-Stenter06 Fabric Length',
        'stenter06': 'TJ-Stenter06 Fabric Length',
        'stenter7': 'TJ-Stenter07 Fabric Length',
        'stenter07': 'TJ-Stenter07 Fabric Length',
        'stenter8': 'TJ-Stenter08 Fabric Length',
        'stenter08': 'TJ-Stenter08 Fabric Length',
        'stenter9': 'TJ-Stenter09 Length',
        'stenter09': 'TJ-Stenter09 Length'
    },
    'energy': {
        'stenter1': 'TJ-Stenter01',
        'stenter01': 'TJ-Stenter01',
        'stenter2': 'TJ-Stenter02',
        'stenter02': 'TJ-Stenter02',
        'stenter3': 'TJ-Stenter03',
        'stenter03': 'TJ-Stenter03',
        'stenter4': 'TJ-Stenter04',
        'stenter04': 'TJ-Stenter04',
        'stenter5': 'TJ-Stenter05',
        'stenter05': 'TJ-Stenter05',
        'stenter6': 'TJ-Stenter06',
        'stenter06': 'TJ-Stenter06',
        'stenter7': 'TJ-Stenter07A',
        'stenter07': 'TJ-Stenter07A',
        'stenter8': 'TJ-Stenter08',
        'stenter08': 'TJ-Stenter08',
        'stenter9': 'TJ-Stenter09',
        'stenter09': 'TJ-Stenter09'
    },
    'utilization': {
        'stenter1': 'TJ-Stenter01 Status',
        'stenter01': 'TJ-Stenter01 Status',
        'stenter2': 'TJ-Stenter02 Status',
        'stenter02': 'TJ-Stenter02 Status',
        'stenter3': 'TJ-Stenter03 Status',
        'stenter03': 'TJ-Stenter03 Status',
        'stenter4': 'TJ-Stenter04 Status',
        'stenter04': 'TJ-Stenter04 Status',
        'stenter5': 'TJ-Stenter05 Status',
        'stenter05': 'TJ-Stenter05 Status',
        'stenter6': 'TJ-Stenter06 Status',
        'stenter06': 'TJ-Stenter06 Status',
        'stenter7': 'TJ-Stenter07 Status',
        'stenter07': 'TJ-Stenter07 Status',
        'stenter8': 'TJ-Stenter08 Status',
        'stenter08': 'TJ-Stenter08 Status',
        'stenter9': 'TJ-Stenter09 Status',
        'stenter09': 'TJ-Stenter09 Status'
    }
}


def detect_query_type(user_query: str):
    """Detect whether the query is about production, energy, or utilization"""
    user_query_lower = user_query.lower()
    
    # Production keywords
    if any(word in user_query_lower for word in ['production', 'length', 'fabric length', 'output', 'produce','length']):
        return 'production'
    
    # Energy keywords  
    elif any(word in user_query_lower for word in ['energy', 'consumption', 'power']):
        return 'energy'
    
    # Utilization keywords
    elif any(word in user_query_lower for word in ['utilization', 'efficiency', 'uptime', 'status', 'downtime']):
        return 'utilization'
    
    # Default to production if unclear
    return 'production'

def extract_machine_numbers(user_query: str):
    """Extract specific machine numbers from user query"""
    import re
    user_query_lower = user_query.lower()
    
    # Look for specific stenter numbers
    stenter_matches = re.findall(r'stenter\s*(\d+)', user_query_lower)
    
    if stenter_matches:
        return [f'stenter{num}' for num in stenter_matches]
    
    # Check for "all machines" or similar
    if any(phrase in user_query_lower for phrase in ['all machines', 'all stenters', 'every machine', 'each machine']):
        return list(MACHINE_MAPPINGS['production'].keys())  # Return all stenter machines
    
    return []


# Enhanced detection function for multi-metric queries
def detect_multi_metric_query(user_query: str):
    """Detect if user is asking for multiple metrics"""
    user_query_lower = user_query.lower()
    
    metrics_mentioned = []
    if any(word in user_query_lower for word in ['production', 'length', 'output', 'produce']):
        metrics_mentioned.append('production')
    
    if any(word in user_query_lower for word in ['energy', 'consumption', 'power']):
        metrics_mentioned.append('energy')
    
    if any(word in user_query_lower for word in ['utilization', 'utilisation', 'efficiency', 'uptime']):
        metrics_mentioned.append('utilization')
    
    return len(metrics_mentioned) > 1, metrics_mentioned

def build_multi_metric_device_names(machine_key: str):
    """Get all device names for a machine across different metrics"""
    return {
        'production_device': MACHINE_MAPPINGS['production'].get(machine_key, ''),
        'energy_device': MACHINE_MAPPINGS['energy'].get(machine_key, ''),
        'utilization_device': MACHINE_MAPPINGS['utilization'].get(machine_key, '')
    }


























def detect_query_intent(user_query: str):
    """Enhanced query intent detection with multi-metric support"""
    user_query_lower = user_query.lower()
    query_type = detect_query_type(user_query)
    
    # Check for multi-metric queries first
    is_multi_metric, metrics = detect_multi_metric_query(user_query)
    
    if is_multi_metric:
        logger.info(f"Multi-metric query detected: {metrics}")
        return 'multi_metric_daily', extract_date_range(user_query), 'multi_metric'
    
    # Single metric queries
    if query_type == 'production':
        if any(word in user_query_lower for word in ['hourly production', 'production by hour', 'hour production']):
            return 'hourly_production', extract_date_range(user_query), query_type
        elif any(word in user_query_lower for word in ['daily production', 'production by day', 'day production', 'production each day']):
            return 'daily_production', extract_date_range(user_query), query_type
    
    # Energy queries
    elif query_type == 'energy':
        if any(word in user_query_lower for word in ['energy consumption', 'hourly energy', 'energy by hour']):
            return 'energy_consumption_hourly', extract_date_range(user_query), query_type
    
    # Utilization queries
    elif query_type == 'utilization':
        if any(word in user_query_lower for word in ['utilization', 'efficiency', 'uptime']):
            return 'utilization_hourly', extract_date_range(user_query), query_type
    
    return None, None, query_type









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
    """Build device filter with proper machine name mapping"""
    query_type = detect_query_type(user_query)
    machine_numbers = extract_machine_numbers(user_query)
    
    logger.info(f"Query type: {query_type}, Machine numbers: {machine_numbers}")
    
    if not machine_numbers:
        return ""  # No specific machines requested
    
    # Get the correct machine names based on query type
    machine_names = []
    for machine in machine_numbers:
        if machine in MACHINE_MAPPINGS[query_type]:
            machine_names.append(MACHINE_MAPPINGS[query_type][machine])
    
    if not machine_names:
        return ""
    
    # Build the WHERE clause
    if len(machine_names) == 1:
        return f"AND device_id = (SELECT virtual_device_id FROM devices WHERE device_name = '{machine_names[0]}')"
    else:
        machine_list = "', '".join(machine_names)
        return f"AND device_id IN (SELECT virtual_device_id FROM devices WHERE device_name IN ('{machine_list}'))"

def get_correct_parameter(query_type: str):
    """Get the correct parameter name based on query type"""
    if query_type == 'production':
        return 'length'
    elif query_type == 'energy':
        return 'TotalEnergy'
    elif query_type == 'utilization':
        return 'status'
    else:
        return 'length'  # default




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
    query_template, _, query_type = detect_query_intent(user_query)


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
    """Enhanced SQL chain with corrected CTE and syntax patterns"""
    template = """
    You are an expert ClickHouse analyst for production monitoring systems. 
    Generate ClickHouse queries that follow the EXACT patterns from the production system queries.
    Generate ClickHouse queries that can handle COMPLEX, MULTI-STEP analysis questions.

    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}

    CRITICAL CLICKHOUSE PRODUCTION QUERY PATTERNS:
    
    1. CTE STRUCTURE (no stray parentheses):
    Always structure WITH clauses cleanly:
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
        SELECT
            device_id,
            hour_bucket,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS production_output
        FROM hourly_windows
        WHERE hour_bucket IS NOT NULL
        GROUP BY device_id, hour_bucket, date
    )
    SELECT 
        dl.device_name AS machine_name,
        pcq.hour_bucket,
        pcq.date,
        pcq.production_output
    FROM production_calculation_query pcq
    LEFT JOIN device_lookup dl ON pcq.device_id = dl.virtual_device_id
    ORDER BY pcq.date, pcq.hour_bucket, dl.device_name
    ```

    2. TIME BUCKET PATTERNS (copy exactly from PDF):
    ```sql
    CASE
        WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 7 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) >= 30 THEN '07:30-08:30'
        WHEN toHour(toTimeZone(timestamp, 'Asia/Colombo')) = 8 AND toMinute(toTimeZone(timestamp, 'Asia/Colombo')) < 30 THEN '07:30-08:30'
        -- Continue all slots exactly as in PDF
        ELSE NULL
    END AS hour_bucket
    ```

    3. PRODUCTION CALCULATION PATTERN:
    ```sql
    argMax(value, timestamp) - argMin(value, timestamp) AS production_output
    ```

    4. ENERGY CONSUMPTION PATTERN:
    Always use `parameter = 'TotalEnergy'`

    5. UTILIZATION PATTERNS:
    ```sql
    sum(CASE WHEN value = 0 THEN interval_duration ELSE 0 END) AS on_time_seconds
    ```

    6. LEAD/LAG FUNCTION RULES:
    Never use NULL defaults. Always provide typed defaults.
    ```sql
    lead(sl_timestamp, 1, toDateTime('2025-04-30 19:30:00', 'Asia/Colombo')) 
        OVER (PARTITION BY device_id ORDER BY sl_timestamp) AS next_timestamp
    ```

    7. UTILIZATION CTE PATTERN:
    ```sql
    WITH status_with_next AS (
        SELECT
            device_id,
            toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
            value,
            toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
            lead(
                toTimeZone(timestamp, 'Asia/Colombo'),
                1,
                toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 19:30:00'), 'Asia/Colombo')
            ) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS next_sl_timestamp
        FROM device_metrics
        WHERE parameter = 'status'
    )
    ```

    9. CRITICAL MACHINE NAME MAPPINGS:
    (same mappings as before: stenter1 → 'TJ-Stenter01 Length(ioid2)', etc.)
    Always use the **full device_name string** in WHERE filters.

    6.MACHINE_NAME_MAPPINGS_FOR_SQL =  ```
        CRITICAL MACHINE NAME MAPPINGS:
    
    When users mention specific stenter machines, you MUST use these exact device names in your WHERE clauses:
    
    For PRODUCTION queries (parameter = 'length'):
    - stenter1 → 'TJ-Stenter01 Length(ioid2)'
    - stenter2 → 'TJ-Stenter02 Length(ioid1)'
    - stenter3 → 'TJ-Stenter03 Length(ioid1)'
    - stenter4 → 'TJ-Stenter04 Length'
    - stenter5 → 'TJ-Stenter05 Length'
    - stenter6 → 'TJ-Stenter06 Fabric Length'
    - stenter7 → 'TJ-Stenter07 Fabric Length'
    - stenter8 → 'TJ-Stenter08 Fabric Length'
    - stenter9 → 'TJ-Stenter09 Length'
    
    For ENERGY queries (parameter = 'TotalEnergy'):
    - stenter1 → 'TJ-Stenter01'
    - stenter2 → 'TJ-Stenter02'
    - stenter3 → 'TJ-Stenter03'
    - stenter4 → 'TJ-Stenter04'
    - stenter5 → 'TJ-Stenter05'
    - stenter6 → 'TJ-Stenter06'
    - stenter7 → 'TJ-Stenter07A'
    - stenter8 → 'TJ-Stenter08'
    - stenter9 → 'TJ-Stenter09'
    
    For UTILIZATION queries (parameter = 'status'):
    - stenter1 → 'TJ-Stenter01 Status'
    - stenter2 → 'TJ-Stenter02 Status'
    - stenter3 → 'TJ-Stenter03 Status'
    - stenter4 → 'TJ-Stenter04 Status'
    - stenter5 → 'TJ-Stenter05 Status'
    - stenter6 → 'TJ-Stenter06 Status'
    - stenter7 → 'TJ-Stenter07 Status'
    - stenter8 → 'TJ-Stenter08 Status'
    - stenter9 → 'TJ-Stenter09 Status'

    REAL PRODUCTION EXAMPLES:
    (include cleaned-up Hourly, Daily, Energy examples as shown earlier)




CRITICAL MULTI-METRIC DEVICE MAPPING:
    
    IMPORTANT: Each physical machine has THREE different device names in the system:
    
    STENTER 1:
    - Production: 'TJ-Stenter01 Length(ioid2)' (parameter = 'length')
    - Energy: 'TJ-Stenter01' (parameter = 'TotalEnergy')  
    - Utilization: 'TJ-Stenter01 Status' (parameter = 'status')
    
    STENTER 2:
    - Production: 'TJ-Stenter02 Length(ioid1)' (parameter = 'length')
    - Energy: 'TJ-Stenter02' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter02 Status' (parameter = 'status')
    
    STENTER 3:
    - Production: 'TJ-Stenter03 Length(ioid1)' (parameter = 'length') 
    - Energy: 'TJ-Stenter03' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter03 Status' (parameter = 'status')
    
    STENTER 4:
    - Production: 'TJ-Stenter04 Length' (parameter = 'length')
    - Energy: 'TJ-Stenter04' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter04 Status' (parameter = 'status')
    
    STENTER 5:
    - Production: 'TJ-Stenter05 Length' (parameter = 'length')
    - Energy: 'TJ-Stenter05' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter05 Status' (parameter = 'status')
    
    STENTER 6:
    - Production: 'TJ-Stenter06 Fabric Length' (parameter = 'length')
    - Energy: 'TJ-Stenter06' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter06 Status' (parameter = 'status')
    
    STENTER 7:
    - Production: 'TJ-Stenter07 Fabric Length' (parameter = 'length')
    - Energy: 'TJ-Stenter07A' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter07 Status' (parameter = 'status')
    
    STENTER 8:
    - Production: 'TJ-Stenter08 Fabric Length' (parameter = 'length')
    - Energy: 'TJ-Stenter08' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter08 Status' (parameter = 'status')
    
    STENTER 9:
    - Production: 'TJ-Stenter09 Length' (parameter = 'length')
    - Energy: 'TJ-Stenter09' (parameter = 'TotalEnergy')
    - Utilization: 'TJ-Stenter09 Status' (parameter = 'status')

    MULTI-METRIC QUERY PATTERNS:

    When users ask for multiple metrics (production, consumption, utilization) for specific dates or machines,
    you MUST create separate CTEs for each metric and join them by machine_name and date.

    EXAMPLE MULTI-METRIC DAILY REPORT PATTERN:
    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    daily_production AS (
        WITH production_raw AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND toDate(toTimeZone(timestamp, 'Asia/Colombo')) = '2025-05-03'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-05-03 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-05-03 19:30:00', 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS daily_production
        FROM production_raw
        GROUP BY device_id, date
    ),
    daily_energy AS (
        WITH energy_raw AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'TotalEnergy'
                AND toDate(toTimeZone(timestamp, 'Asia/Colombo')) = '2025-05-03'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-05-03 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-05-03 19:30:00', 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS daily_consumption
        FROM energy_raw
        GROUP BY device_id, date
    ),
    daily_utilization AS (
        WITH status_with_next AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value AS status_value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
                lead(
                    toTimeZone(timestamp, 'Asia/Colombo'),
                    1,
                    toDateTime('2025-05-03 19:30:00', 'Asia/Colombo')
                ) OVER (PARTITION BY device_id ORDER BY timestamp) AS next_sl_timestamp
            FROM device_metrics
            WHERE parameter = 'status'
                AND toDate(toTimeZone(timestamp, 'Asia/Colombo')) = '2025-05-03'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-05-03 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-05-03 19:30:00', 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) AS on_time_seconds,
            sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) AS total_seconds,
            CASE 
                WHEN sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) > 0 
                THEN (sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) * 100.0) / sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)))
                ELSE 0 
            END AS daily_utilization_percentage
        FROM status_with_next
        GROUP BY device_id, date
    ),
    -- Create machine mapping to consolidate different device_ids for same physical machine
    machine_consolidation AS (
        SELECT 
            -- Extract machine number from device name to group same machines
            CASE 
                WHEN dl.device_name LIKE '%Stenter01%' THEN 'Stenter 1'
                WHEN dl.device_name LIKE '%Stenter02%' THEN 'Stenter 2'  
                WHEN dl.device_name LIKE '%Stenter03%' THEN 'Stenter 3'
                WHEN dl.device_name LIKE '%Stenter04%' THEN 'Stenter 4'
                WHEN dl.device_name LIKE '%Stenter05%' THEN 'Stenter 5'
                WHEN dl.device_name LIKE '%Stenter06%' THEN 'Stenter 6'
                WHEN dl.device_name LIKE '%Stenter07%' THEN 'Stenter 7'
                WHEN dl.device_name LIKE '%Stenter08%' THEN 'Stenter 8'
                WHEN dl.device_name LIKE '%Stenter09%' THEN 'Stenter 9'
                ELSE dl.device_name
            END AS machine_name,
            dl.virtual_device_id,
            dl.device_name,
            '2025-05-03' AS report_date
        FROM device_lookup dl
    )
    SELECT 
        mc.machine_name,
        mc.report_date AS date,
        COALESCE(dp.daily_production, 0) AS production_output,
        COALESCE(de.daily_consumption, 0) AS energy_consumption, 
        COALESCE(du.daily_utilization_percentage, 0) AS utilization_percentage
    FROM machine_consolidation mc
    LEFT JOIN daily_production dp ON mc.virtual_device_id = dp.device_id
    LEFT JOIN daily_energy de ON mc.virtual_device_id = de.device_id
    LEFT JOIN daily_utilization du ON mc.virtual_device_id = du.device_id
    WHERE mc.machine_name LIKE 'Stenter%'  -- Only include Stenter machines
    ORDER BY mc.machine_name
    ```

    FILTERED MULTI-METRIC PATTERN:
    For queries like "Which days had less than 50% utilization and what was production/consumption":

    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    -- Step 1: Calculate utilization for filtering
    daily_utilization AS (
        WITH status_with_next AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value AS status_value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
                lead(
                    toTimeZone(timestamp, 'Asia/Colombo'),
                    1,
                    toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 19:30:00'), 'Asia/Colombo')
                ) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS next_sl_timestamp
            FROM device_metrics
            WHERE parameter = 'status'
                AND toDate(toTimeZone(timestamp, 'Asia/Colombo')) >= '2025-04-01'
                AND toDate(toTimeZone(timestamp, 'Asia/Colombo')) <= '2025-04-30'
        )
        SELECT
            device_id,
            date,
            CASE 
                WHEN sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) > 0 
                THEN (sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) * 100.0) / sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)))
                ELSE 0 
            END AS daily_utilization_percentage
        FROM status_with_next
        GROUP BY device_id, date
    ),
    -- Step 2: Filter days with low utilization
    filtered_days AS (
        SELECT 
            device_id,
            date,
            daily_utilization_percentage
        FROM daily_utilization
        WHERE daily_utilization_percentage < 50.0
    ),
    -- Step 3: Get production for filtered days only
    filtered_production AS (
        WITH production_raw AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND (device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo'))) IN (
                    SELECT device_id, date FROM filtered_days
                )
        )
        SELECT
            device_id,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS daily_production
        FROM production_raw
        GROUP BY device_id, date
    ),
    -- Step 4: Get energy for filtered days only  
    filtered_energy AS (
        WITH energy_raw AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'TotalEnergy'
                AND (device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo'))) IN (
                    SELECT device_id, date FROM filtered_days
                )
        )
        SELECT
            device_id,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS daily_consumption
        FROM energy_raw
        GROUP BY device_id, date
    ),
    -- Step 5: Consolidate by machine name
    machine_consolidation AS (
        SELECT 
            CASE 
                WHEN dl.device_name LIKE '%Stenter01%' THEN 'Stenter 1'
                WHEN dl.device_name LIKE '%Stenter02%' THEN 'Stenter 2'
                WHEN dl.device_name LIKE '%Stenter03%' THEN 'Stenter 3'
                WHEN dl.device_name LIKE '%Stenter04%' THEN 'Stenter 4'
                WHEN dl.device_name LIKE '%Stenter05%' THEN 'Stenter 5'
                WHEN dl.device_name LIKE '%Stenter06%' THEN 'Stenter 6'
                WHEN dl.device_name LIKE '%Stenter07%' THEN 'Stenter 7'
                WHEN dl.device_name LIKE '%Stenter08%' THEN 'Stenter 8'
                WHEN dl.device_name LIKE '%Stenter09%' THEN 'Stenter 9'
                ELSE dl.device_name
            END AS machine_name,
            dl.virtual_device_id,
            dl.device_name
        FROM device_lookup dl
    )
    SELECT DISTINCT
        mc.machine_name,
        fd.date,
        fd.daily_utilization_percentage AS utilization_percentage,
        COALESCE(fp.daily_production, 0) AS production_output,
        COALESCE(fe.daily_consumption, 0) AS energy_consumption
    FROM filtered_days fd
    LEFT JOIN machine_consolidation mc ON fd.device_id = mc.virtual_device_id
    LEFT JOIN filtered_production fp ON fd.device_id = fp.device_id AND fd.date = fp.date
    LEFT JOIN filtered_energy fe ON fd.device_id = fe.device_id AND fd.date = fe.date  
    WHERE mc.machine_name IS NOT NULL
    ORDER BY fd.date, mc.machine_name
    ```

    CRITICAL RULES FOR MULTI-METRIC QUERIES:

    1. **ALWAYS CREATE SEPARATE CTEs FOR EACH METRIC**:
       - daily_production CTE with parameter = 'length'
       - daily_energy CTE with parameter = 'TotalEnergy'  
       - daily_utilization CTE with parameter = 'status'

    2. **USE MACHINE CONSOLIDATION**:
       - Create machine_consolidation CTE to map device names to common machine names
       - Use LIKE patterns to identify same physical machines
       - Group by the consolidated machine_name in final results

    3. **PROPER JOIN STRATEGY**:
       - Start with the primary filtering condition (dates, machines, thresholds)
       - LEFT JOIN all metric CTEs to ensure no data loss
       - Always use COALESCE(metric_value, 0) for missing values

    4. **HANDLE DIFFERENT DEVICE IDS**:
       - Understand that same machine has different device_id for different metrics
       - Use device name patterns (Stenter01, Stenter02, etc.) to identify same machines
       - Join metrics by matching machine patterns, not just device_id

    5. **DATE FILTERING**:
       - Apply date filters consistently across all metric CTEs
       - Use same timezone ('Asia/Colombo') and time ranges (07:30-19:30) for all metrics

    6. **SPECIFIC MACHINE FILTERING**:
       When user mentions specific machines (e.g., "stenter 4"), apply filters like:
       ```sql
       WHERE dl.device_name LIKE '%Stenter04%'
       ```
       This will match all three device types for that machine.

























    COMPLEX QUERY PATTERNS AND EXAMPLES:

    1. MULTI-STEP ANALYSIS PATTERN:
    For questions like "What is the highest utilization in March and production of that day?":
    
    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    daily_utilization AS (
        WITH status_with_next AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value AS status_value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
                lead(
                    toTimeZone(timestamp, 'Asia/Colombo'),
                    1,
                    toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 19:30:00'), 'Asia/Colombo')
                ) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS next_sl_timestamp
            FROM device_metrics
            WHERE parameter = 'status'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-03-01 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-03-31 19:30:00', 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) AS on_time_seconds,
            sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) AS total_seconds,
            CASE 
                WHEN sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) > 0 
                THEN (sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) * 100.0) / sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)))
                ELSE 0 
            END AS daily_utilization_percentage
        FROM status_with_next
        WHERE sl_timestamp >= toDateTime(concat(toString(date), ' 07:30:00'), 'Asia/Colombo')
            AND sl_timestamp <= toDateTime(concat(toString(date), ' 19:30:00'), 'Asia/Colombo')
        GROUP BY device_id, date
    ),
    max_utilization_day AS (
        SELECT 
            date AS target_date,
            max(daily_utilization_percentage) AS max_utilization
        FROM daily_utilization
        GROUP BY date
        ORDER BY max_utilization DESC
        LIMIT 1
    ),
    production_for_target_day AS (
        WITH hourly_windows AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND toDate(toTimeZone(timestamp, 'Asia/Colombo')) = (SELECT target_date FROM max_utilization_day)
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 07:30:00'), 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 19:30:00'), 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS daily_production
        FROM hourly_windows
        GROUP BY device_id, date
    )
    SELECT 
        dl.device_name AS machine_name,
        mut.target_date AS analysis_date,
        mut.max_utilization AS highest_utilization_percentage,
        COALESCE(pft.daily_production, 0) AS production_on_that_day,
        du.daily_utilization_percentage AS machine_specific_utilization
    FROM max_utilization_day mut
    CROSS JOIN device_lookup dl
    LEFT JOIN daily_utilization du ON dl.virtual_device_id = du.device_id AND du.date = mut.target_date
    LEFT JOIN production_for_target_day pft ON dl.virtual_device_id = pft.device_id
    WHERE du.daily_utilization_percentage IS NOT NULL
    ORDER BY du.daily_utilization_percentage DESC
    ```

    2. THRESHOLD-BASED FILTERING PATTERN:
    For questions like "Which days in April had less than 50% utilization?":
    
    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    daily_utilization AS (
        -- [Same utilization calculation CTE as above]
    ),
    low_utilization_days AS (
        SELECT 
            device_id,
            date,
            daily_utilization_percentage
        FROM daily_utilization
        WHERE daily_utilization_percentage < 50.0
            AND date >= toDate('2025-04-01')
            AND date <= toDate('2025-04-30')
    ),
    production_for_low_days AS (
        WITH hourly_windows AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND toDate(toTimeZone(timestamp, 'Asia/Colombo')) IN (
                    SELECT DISTINCT date FROM low_utilization_days
                )
        )
        SELECT
            device_id,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS daily_production
        FROM hourly_windows
        GROUP BY device_id, date
    )
    SELECT 
        dl.device_name AS machine_name,
        lud.date AS low_utilization_date,
        lud.daily_utilization_percentage AS utilization_percentage,
        COALESCE(pld.daily_production, 0) AS production_output
    FROM low_utilization_days lud
    LEFT JOIN device_lookup dl ON lud.device_id = dl.virtual_device_id
    LEFT JOIN production_for_low_days pld ON lud.device_id = pld.device_id AND lud.date = pld.date
    ORDER BY lud.date, lud.daily_utilization_percentage ASC
    ```

    3. COMPARISON AND RANKING PATTERNS:
    For questions involving "best performing", "worst day", "top 5", etc.:
    
    ```sql
    WITH ranked_performance AS (
        -- Calculate metrics first
        SELECT 
            device_id,
            date,
            metric_value,
            row_number() OVER (ORDER BY metric_value DESC) AS performance_rank
        FROM your_calculation_cte
    )
    SELECT * FROM ranked_performance WHERE performance_rank <= 5
    ```

    4. TIME-BASED AGGREGATION WITH CONDITIONS:
    For questions like "Average production on days with high utilization":
    
    ```sql
    WITH condition_filter AS (
        SELECT device_id, date 
        FROM utilization_cte 
        WHERE utilization_percentage > 80.0
    ),
    conditional_production AS (
        SELECT 
            cf.device_id,
            cf.date,
            production_calculation
        FROM condition_filter cf
        JOIN production_cte pc ON cf.device_id = pc.device_id AND cf.date = pc.date
    )
    SELECT 
        device_name,
        avg(production_calculation) AS avg_production_on_high_util_days
    FROM conditional_production
    GROUP BY device_id
    ```

    CRITICAL RULES FOR COMPLEX QUERIES:

    1. **ALWAYS BREAK DOWN COMPLEX QUESTIONS**: 
       - Identify all the components (what metrics, what conditions, what comparisons)
       - Create separate CTEs for each major calculation step
       - Join results together in the final SELECT

    2. **USE SUBQUERIES IN WHERE CLAUSES**:
       - When filtering by "days that meet condition X", use IN (SELECT date FROM condition_cte)
       - When finding "the day with max/min value", use subqueries with LIMIT 1

    3. **HANDLE MULTIPLE METRICS PROPERLY**:
       - If question asks for both utilization AND production, calculate both
       - Use LEFT JOINs to ensure you don't lose data
       - Always use COALESCE for nullable results

    4. **RANKING AND FILTERING**:
       - Use row_number(), rank(), or dense_rank() for "top N" or "best/worst" questions
       - Use HAVING clause for group-level filtering
       - Use WHERE clause for row-level filtering

    5. **DATE RANGE HANDLING**:
       - Always extract date ranges from the question
       - Apply date filters early in CTEs for performance
       - Use proper date functions: toDate(), toDateTime()

    6. **AGGREGATION LEVELS**:
       - Daily level: GROUP BY device_id, date
       - Machine level: GROUP BY device_id
       - Overall level: No GROUP BY or GROUP BY () for totals













        CRITICAL PATTERN FOR CONDITIONAL DATA RETRIEVAL:

    When users ask: "Which days had [condition] and what was [other metric] on those days?"
    
    YOU MUST USE THIS EXACT PATTERN:

    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    -- STEP 1: Calculate the filtering metric (e.g., utilization)
    daily_utilization AS (
        WITH status_with_next AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value AS status_value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
                lead(
                    toTimeZone(timestamp, 'Asia/Colombo'),
                    1,
                    toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 19:30:00'), 'Asia/Colombo')
                ) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS next_sl_timestamp
            FROM device_metrics
            WHERE parameter = 'status'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-05-01 07:30:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-05-31 19:30:00', 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) AS on_time_seconds,
            sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) AS total_seconds,
            CASE 
                WHEN sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) > 0 
                THEN (sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) * 100.0) / sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)))
                ELSE 0 
            END AS daily_utilization_percentage
        FROM status_with_next
        WHERE sl_timestamp >= toDateTime(concat(toString(date), ' 07:30:00'), 'Asia/Colombo')
            AND sl_timestamp <= toDateTime(concat(toString(date), ' 19:30:00'), 'Asia/Colombo')
        GROUP BY device_id, date
    ),
    -- STEP 2: Filter days that meet the condition
    filtered_days AS (
        SELECT 
            device_id,
            date,
            daily_utilization_percentage
        FROM daily_utilization
        WHERE daily_utilization_percentage > 50.0  -- APPLY THE CONDITION HERE
    ),
    -- STEP 3: Calculate the requested metric ONLY for filtered days
    production_for_filtered_days AS (
        WITH hourly_windows AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date
            FROM device_metrics
            WHERE parameter = 'length'
                AND (device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo'))) IN (
                    SELECT device_id, date FROM filtered_days  -- CRITICAL: Only get production for filtered days
                )
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 07:30:00'), 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 19:30:00'), 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            argMax(value, sl_timestamp) - argMin(value, sl_timestamp) AS daily_production
        FROM hourly_windows
        GROUP BY device_id, date
    )
    -- STEP 4: Join everything together
    SELECT 
        dl.device_name AS machine_name,
        fd.date AS filtered_date,
        fd.daily_utilization_percentage AS utilization_percentage,
        COALESCE(pfd.daily_production, 0) AS production_output
    FROM filtered_days fd
    LEFT JOIN device_lookup dl ON fd.device_id = dl.virtual_device_id
    LEFT JOIN production_for_filtered_days pfd ON fd.device_id = pfd.device_id AND fd.date = pfd.date
    ORDER BY fd.date, dl.device_name
    ```

    MANDATORY RULES FOR CONDITIONAL QUERIES:

    1. **ALWAYS CREATE A FILTERED_DAYS CTE**: 
       - This contains ONLY the days/machines that meet your condition
       - Apply the condition filter here (>, <, =, BETWEEN, etc.)

    2. **USE FILTERED_DAYS IN SUBSEQUENT QUERIES**:
       - When calculating other metrics, use: WHERE (device_id, date) IN (SELECT device_id, date FROM filtered_days)
       - This ensures you ONLY calculate metrics for the days that met your condition

    3. **PROPER JOIN STRUCTURE**:
       - Start FROM the filtered_days CTE
       - LEFT JOIN other metrics to it
       - This ensures you get a row for every filtered day, even if other metrics are missing

    4. **HANDLE MISSING DATA**:
       - Always use COALESCE(metric_value, 0) in final SELECT
       - Use LEFT JOIN, not INNER JOIN, to preserve filtered days

    5. **CONDITION MAPPING**:
       - "greater than 50%" → WHERE metric > 50.0
       - "less than 50%" → WHERE metric < 50.0  
       - "between 40 and 60%" → WHERE metric BETWEEN 40.0 AND 60.0
       - "exactly 80%" → WHERE metric = 80.0

    SPECIFIC EXAMPLES FOR COMMON CONDITIONAL PATTERNS:

    A) "Days with low utilization and their production":
    ```sql
    filtered_days AS (
        SELECT device_id, date, daily_utilization_percentage
        FROM daily_utilization
        WHERE daily_utilization_percentage < 50.0
    )
    ```

    B) "Days with high production and their utilization":
    ```sql
    filtered_days AS (
        SELECT device_id, date, daily_production
        FROM daily_production_cte
        WHERE daily_production > 1000.0
    )
    ```

    C) "Best performing days" (top 10):
    ```sql
    filtered_days AS (
        SELECT device_id, date, daily_utilization_percentage,
               row_number() OVER (PARTITION BY device_id ORDER BY daily_utilization_percentage DESC) as rank_num
        FROM daily_utilization
    ),
    top_days AS (
        SELECT device_id, date, daily_utilization_percentage
        FROM filtered_days 
        WHERE rank_num <= 10
    )




    CRITICAL COUNTER CALCULATION PATTERNS:

    PROBLEM: Many production systems have counters that RESET daily, during maintenance, or have irregular patterns.
    The simple argMax(value) - argMin(value) often returns 0 when counters reset.

    SOLUTION: Use ROBUST counter calculation with multiple fallback methods:

    ```sql
    WITH device_lookup AS (
        SELECT virtual_device_id, device_name FROM devices
    ),
    -- STEP 1: Get raw data with debugging info
    raw_production_data AS (
        SELECT
            device_id,
            toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
            value,
            toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
            lag(value, 1, 0) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS prev_value,
            value - lag(value, 1, 0) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS increment
        FROM device_metrics
        WHERE parameter = 'length'
            AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-05-01 00:00:00', 'Asia/Colombo')
            AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-05-31 23:59:59', 'Asia/Colombo')
    ),
    -- STEP 2: Calculate production using MULTIPLE methods
    daily_production_analysis AS (
        SELECT
            device_id,
            date,
            count(*) as record_count,
            min(value) as min_value,
            max(value) as max_value,
            max(value) - min(value) as simple_diff,
            -- Method 1: Sum positive increments (handles counter resets)
            sum(CASE WHEN increment > 0 AND increment < 10000 THEN increment ELSE 0 END) as sum_increments,
            -- Method 2: Sum all positive values (for rate-based data)
            sum(CASE WHEN value > 0 AND value < 1000 THEN value ELSE 0 END) as sum_values,
            -- Method 3: Average rate * time (for continuous production)
            avg(CASE WHEN value > 0 THEN value ELSE NULL END) as avg_rate
        FROM raw_production_data
        GROUP BY device_id, date
    ),
    -- STEP 3: Choose best calculation method
    daily_production AS (
        SELECT
            device_id,
            date,
            record_count,
            min_value,
            max_value,
            -- Use the method that gives reasonable results
            CASE 
                WHEN sum_increments > 0 THEN sum_increments
                WHEN simple_diff > 0 AND simple_diff < 50000 THEN simple_diff
                WHEN sum_values > 0 THEN sum_values
                WHEN avg_rate > 0 THEN avg_rate * 12  -- Assuming 12-hour work day
                ELSE 0
            END as daily_production,
            -- Debug info
            sum_increments,
            simple_diff,
            sum_values,
            avg_rate
        FROM daily_production_analysis
    ),
    -- STEP 4: Similar robust calculation for energy
    raw_energy_data AS (
        SELECT
            device_id,
            toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
            value,
            toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
            lag(value, 1, 0) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS prev_value,
            value - lag(value, 1, 0) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS increment
        FROM device_metrics
        WHERE parameter = 'TotalEnergy'
            AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-05-01 00:00:00', 'Asia/Colombo')
            AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-05-31 23:59:59', 'Asia/Colombo')
    ),
    daily_energy AS (
        SELECT
            device_id,
            date,
            count(*) as record_count,
            CASE 
                WHEN sum(CASE WHEN increment > 0 AND increment < 10000 THEN increment ELSE 0 END) > 0 
                THEN sum(CASE WHEN increment > 0 AND increment < 10000 THEN increment ELSE 0 END)
                WHEN max(value) - min(value) > 0 AND max(value) - min(value) < 100000 
                THEN max(value) - min(value)
                WHEN sum(CASE WHEN value > 0 AND value < 5000 THEN value ELSE 0 END) > 0 
                THEN sum(CASE WHEN value > 0 AND value < 5000 THEN value ELSE 0 END)
                ELSE 0
            END as daily_consumption
        FROM raw_energy_data
        GROUP BY device_id, date
    ),
    -- STEP 5: Utilization calculation (keep existing logic)
    daily_utilization AS (
        WITH status_with_next AS (
            SELECT
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value AS status_value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
                lead(
                    toTimeZone(timestamp, 'Asia/Colombo'),
                    1,
                    toDateTime(concat(toString(toDate(toTimeZone(timestamp, 'Asia/Colombo'))), ' 23:59:59'), 'Asia/Colombo')
                ) OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS next_sl_timestamp
            FROM device_metrics
            WHERE parameter = 'status'
                AND toTimeZone(timestamp, 'Asia/Colombo') >= toDateTime('2025-05-01 00:00:00', 'Asia/Colombo')
                AND toTimeZone(timestamp, 'Asia/Colombo') <= toDateTime('2025-05-31 23:59:59', 'Asia/Colombo')
        )
        SELECT
            device_id,
            date,
            sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) AS on_time_seconds,
            sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) AS total_seconds,
            CASE 
                WHEN sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp))) > 0 
                THEN (sum(CASE WHEN status_value = 0 THEN greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)) ELSE 0 END) * 100.0) / sum(greatest(0, dateDiff('second', sl_timestamp, next_sl_timestamp)))
                ELSE 0 
            END AS daily_utilization_percentage
        FROM status_with_next
        GROUP BY device_id, date
    ),
    -- STEP 6: Apply filtering condition
    filtered_days AS (
        SELECT 
            device_id,
            date,
            daily_utilization_percentage
        FROM daily_utilization
        WHERE daily_utilization_percentage > 50.0
    )
    -- STEP 7: Final result with debugging info
    SELECT 
        dl.device_name AS machine_name,
        fd.date AS filtered_date,
        fd.daily_utilization_percentage AS utilization_percentage,
        COALESCE(dp.daily_production, 0) AS production_output,
        COALESCE(de.daily_consumption, 0) AS energy_consumption,
        -- Debug columns (remove these in production)
        dp.record_count AS prod_records,
        dp.min_value AS prod_min,
        dp.max_value AS prod_max,
        dp.simple_diff AS prod_simple_diff,
        dp.sum_increments AS prod_sum_increments,
        de.record_count AS energy_records
    FROM filtered_days fd
    LEFT JOIN device_lookup dl ON fd.device_id = dl.virtual_device_id
    LEFT JOIN daily_production dp ON fd.device_id = dp.device_id AND fd.date = dp.date
    LEFT JOIN daily_energy de ON fd.device_id = de.device_id AND fd.date = de.date
    ORDER BY fd.date, dl.device_name
    ```

    ALTERNATIVE PATTERN FOR NON-COUNTER DATA:
    If your data is stored as rates or instantaneous values instead of cumulative counters:

    ```sql
    -- For rate-based production data (meters/hour, units/hour, etc.)
    daily_production AS (
        SELECT
            device_id,
            date,
            -- Sum all positive rates * time intervals
            sum(CASE WHEN value > 0 THEN value * (dateDiff('second', sl_timestamp, next_sl_timestamp) / 3600.0) ELSE 0 END) as daily_production
        FROM (
            SELECT 
                device_id,
                toTimeZone(timestamp, 'Asia/Colombo') AS sl_timestamp,
                value,
                toDate(toTimeZone(timestamp, 'Asia/Colombo')) AS date,
                lead(toTimeZone(timestamp, 'Asia/Colombo'), 1, toDateTime('2025-05-31 23:59:59', 'Asia/Colombo')) 
                    OVER (PARTITION BY device_id, toDate(toTimeZone(timestamp, 'Asia/Colombo')) ORDER BY timestamp) AS next_sl_timestamp
            FROM device_metrics
            WHERE parameter = 'length'
        )
        GROUP BY device_id, date
    )
    ```

    PARAMETER NAME DEBUGGING:
    If production/energy are still 0, first check what parameters exist:

    ```sql
    -- Debug query to find available parameters
    SELECT 
        parameter,
        count(*) as record_count,
        min(value) as min_val,
        max(value) as max_val,
        avg(value) as avg_val
    FROM device_metrics 
    WHERE toDate(timestamp) = '2025-05-01'
    GROUP BY parameter
    ORDER BY record_count DESC
    ```

    DEVICE NAME VERIFICATION:
    Check actual device names in your database:

    ```sql
    -- Debug query to find device names containing production parameters
    SELECT DISTINCT 
        d.device_name,
        dm.parameter,
        count(*) as records
    FROM devices d
    JOIN device_metrics dm ON d.virtual_device_id = dm.device_id
    WHERE toDate(dm.timestamp) = '2025-05-01'
        AND (dm.parameter LIKE '%length%' OR dm.parameter LIKE '%production%' OR dm.parameter LIKE '%energy%')
    GROUP BY d.device_name, dm.parameter
    ORDER BY d.device_name, dm.parameter
    ```












    

    MANDATORY REQUIREMENTS:
    1. Always use 'Asia/Colombo' timezone
    2. Always filter working hours: 07:30:00–19:30:00
    3. Always use correct parameters: 'length', 'TotalEnergy', 'status'
    4. Always use device_lookup CTE
    5. Always use argMax/argMin (never SUM for counters)
    6. Always provide typed defaults in window functions
    7. Always ensure parentheses are balanced (no extra `)`)
    8. Always order final results by date, time_bucket, machine_name
    9. Always wrap time differences in greatest(0, …) for utilization
    10. Write ONLY the SQL query. No backticks, no markdown.

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
        
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0,
        max_tokens=30000,
    )
        
    def get_schema(_):
        return """
        Tables:
        - devices: virtual_device_id (Int64), device_name (String)
        - device_metrics: device_id (Int64), timestamp (DateTime), parameter (String), value (Float64)
        
        Parameters:
        - 'length': Production counter values
        - 'TotalEnergy': Energy consumption counter values
        - 'status': Machine status (0=ON, 1=OFF)
        
        Time Zones:
        - Always use 'Asia/Colombo'
        - Working hours: 07:30:00–19:30:00
        """
        
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

    
    














    
    
    
    




def create_enhanced_visualization(df, chart_type, user_query):
    """Enhanced visualization with proper temporal granularity detection and axis handling"""
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
        
        # Enhanced column detection with better patterns
        def detect_columns(df):
            """Detect time, machine, and value columns with improved logic"""
            time_col = None
            machine_col = None
            value_col = None
            granularity = 'unknown'
            
            # Detect time column and granularity
            time_patterns = {
                'hourly': ['hour', 'time_hour', 'production_hour', 'hourly', 'hr'],
                'daily': ['day', 'date', 'production_date', 'daily', 'production_day'],
                'monthly': ['month', 'production_month', 'monthly', 'mon']
            }
            
            for gran, patterns in time_patterns.items():
                for col in df.columns:
                    col_lower = col.lower()
                    if any(pattern in col_lower for pattern in patterns):
                        time_col = col
                        granularity = gran
                        logger.info(f"Detected {granularity} time column: {col}")
                        break
                if time_col:
                    break
            
            # If no explicit time column found, check for datetime columns
            if not time_col:
                for col in df.columns:
                    if df[col].dtype == 'datetime64[ns]' or col.lower() in ['timestamp', 'datetime']:
                        time_col = col
                        # Try to infer granularity from data
                        if len(df) > 1:
                            time_diff = df[col].diff().dropna()
                            if not time_diff.empty:
                                median_diff = time_diff.median()
                                if median_diff <= pd.Timedelta(hours=2):
                                    granularity = 'hourly'
                                elif median_diff <= pd.Timedelta(days=2):
                                    granularity = 'daily'
                                else:
                                    granularity = 'monthly'
                        break
            
            # Detect machine column
            machine_patterns = ['machine', 'device', 'equipment', 'unit', 'station']
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in machine_patterns):
                    machine_col = col
                    logger.info(f"Detected machine column: {col}")
                    break
            
            # Detect value column based on granularity and metric type
            value_patterns = {
                'production': ['production', 'output', 'manufactured', 'produced'],
                'consumption': ['consumption', 'consumed', 'usage', 'used'],
                'utilisation': ['utilisation', 'utilization', 'efficiency', 'usage_rate']
            }
            
            # First try to match with granularity prefix
            for metric, patterns in value_patterns.items():
                for col in df.columns:
                    col_lower = col.lower()
                    # Check for granularity-specific columns (e.g., hourly_production, daily_consumption)
                    if f"{granularity}_{metric}" in col_lower or f"{metric}_{granularity}" in col_lower:
                        value_col = col
                        logger.info(f"Detected {granularity} {metric} column: {col}")
                        break
                    # Check for general patterns with granularity
                    elif granularity in col_lower and any(pattern in col_lower for pattern in patterns):
                        value_col = col
                        logger.info(f"Detected {granularity} {metric} column: {col}")
                        break
                if value_col:
                    break
            
            # Fallback to general value column detection
            if not value_col:
                for metric, patterns in value_patterns.items():
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(pattern in col_lower for pattern in patterns):
                            value_col = col
                            logger.info(f"Detected general {metric} column: {col}")
                            break
                    if value_col:
                        break
            
            # Last fallback to first numeric column
            if not value_col and numeric_cols:
                value_col = numeric_cols[0]
                logger.info(f"Using fallback numeric column: {value_col}")
            
            return time_col, machine_col, value_col, granularity
        
        # Detect columns
        time_col, machine_col, value_col, granularity = detect_columns(df)
        
        logger.info(f"Final detection - Time: {time_col}, Machine: {machine_col}, Value: {value_col}, Granularity: {granularity}")
        
        # Override chart type based on user query
        if user_query:
            query_lower = user_query.lower()
            if "bar chart" in query_lower or "bar" in query_lower:
                chart_type = "bar"
            elif "pie chart" in query_lower or "pie" in query_lower:
                chart_type = "pie"
            elif "line chart" in query_lower or "line" in query_lower:
                chart_type = "line"
        
        fig = None
        
        # Prepare data based on granularity
        def prepare_data_for_visualization(df, time_col, machine_col, value_col, granularity):
            """Prepare and group data appropriately for visualization"""
            viz_df = df.copy()
            
            # Handle datetime columns
            if time_col and df[time_col].dtype == 'datetime64[ns]':
                if granularity == 'hourly':
                    viz_df['time_display'] = viz_df[time_col].dt.hour
                    time_label = "Hour of Day"
                elif granularity == 'daily':
                    viz_df['time_display'] = viz_df[time_col].dt.date
                    time_label = "Date"
                elif granularity == 'monthly':
                    viz_df['time_display'] = viz_df[time_col].dt.to_period('M').astype(str)
                    time_label = "Month"
                else:
                    viz_df['time_display'] = viz_df[time_col]
                    time_label = "Time"
                
                time_col = 'time_display'
            else:
                # For non-datetime columns, use as-is
                if granularity == 'hourly':
                    time_label = "Hour"
                elif granularity == 'daily':
                    time_label = "Day"
                elif granularity == 'monthly':
                    time_label = "Month"
                else:
                    time_label = time_col.replace('_', ' ').title() if time_col else "Time"
            
            # Group data to avoid duplicates
            if time_col and machine_col and value_col:
                viz_df = viz_df.groupby([time_col, machine_col])[value_col].sum().reset_index()
            elif time_col and value_col:
                viz_df = viz_df.groupby(time_col)[value_col].sum().reset_index()
            
            return viz_df, time_col, time_label
        
        # Create appropriate chart title and labels
        def create_chart_labels(chart_type, granularity, value_col, machine_col):
            """Create appropriate chart titles and axis labels"""
            # Determine metric type
            metric_type = "Value"
            if value_col:
                value_lower = value_col.lower()
                if 'production' in value_lower:
                    metric_type = "Production"
                elif 'consumption' in value_lower:
                    metric_type = "Consumption"
                elif 'utilisation' in value_lower or 'utilization' in value_lower:
                    metric_type = "Utilisation"
            
            # Create title
            granularity_title = granularity.title() if granularity != 'unknown' else ""
            title = f"{granularity_title} {metric_type} Analysis"
            
            # Create Y-axis label
            y_label = f"{granularity_title} {metric_type}" if granularity != 'unknown' else value_col.replace('_', ' ').title()
            
            return title, y_label, metric_type
        
        # Prepare data
        viz_df, time_col, time_label = prepare_data_for_visualization(df, time_col, machine_col, value_col, granularity)
        title, y_label, metric_type = create_chart_labels(chart_type, granularity, value_col, machine_col)
        
        # Create visualizations based on chart type
        if chart_type == "bar":
            if time_col and value_col:
                if machine_col and viz_df[machine_col].nunique() > 1:
                    # Grouped bar chart by machine
                    fig = px.bar(
                        viz_df, 
                        x=time_col, 
                        y=value_col, 
                        color=machine_col,
                        title=title,
                        labels={
                            time_col: time_label,
                            value_col: y_label,
                            machine_col: "Machine"
                        },
                        barmode='group'
                    )
                else:
                    # Simple bar chart
                    fig = px.bar(
                        viz_df, 
                        x=time_col, 
                        y=value_col,
                        title=title,
                        labels={
                            time_col: time_label,
                            value_col: y_label
                        }
                    )
        
        elif chart_type == "line":
            if time_col and value_col:
                fig = px.line(
                    viz_df, 
                    x=time_col, 
                    y=value_col, 
                    color=machine_col if machine_col else None,
                    title=title,
                    labels={
                        time_col: time_label,
                        value_col: y_label,
                        machine_col: "Machine" if machine_col else None
                    },
                    markers=True
                )
        
        elif chart_type == "pie":
            if machine_col and value_col:
                # Pie chart showing distribution by machine
                pie_df = viz_df.groupby(machine_col)[value_col].sum().reset_index()
                fig = px.pie(
                    pie_df, 
                    names=machine_col, 
                    values=value_col,
                    title=f"{metric_type} Distribution by Machine"
                )
            elif time_col and value_col and len(viz_df) <= 20:  # Avoid too many slices
                fig = px.pie(
                    viz_df, 
                    names=time_col, 
                    values=value_col,
                    title=f"{title} Distribution"
                )
        
        # Format axes based on granularity
        if fig and granularity == 'hourly' and time_col:
            fig.update_xaxes(
                tickmode='linear',
                tick0=0,
                dtick=1 if granularity == 'hourly' else None,
                tickangle=0 if granularity == 'hourly' else 45
            )
        elif fig and granularity in ['daily', 'monthly']:
            fig.update_xaxes(tickangle=45)
        
        # Enhanced layout
        if fig:
            fig.update_layout(
                showlegend=True,
                height=600,
                margin=dict(l=60, r=50, t=80, b=100),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                title_font_size=16,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            logger.info("Visualization created successfully")
            
            # Enhanced data summary
            with st.expander("📊 Data Summary"):
                st.write(f"**Chart Type:** {chart_type.title()}")
                st.write(f"**Time Granularity:** {granularity.title()}")
                st.write(f"**Metric Type:** {metric_type}")
                st.write(f"**Total Records:** {len(viz_df)}")
                
                if machine_col and machine_col in viz_df.columns:
                    unique_machines = viz_df[machine_col].nunique()
                    st.write(f"**Unique Machines:** {unique_machines}")
                    st.write(f"**Machines:** {', '.join(viz_df[machine_col].unique()[:5])}")
                    if unique_machines > 5:
                        st.write(f"... and {unique_machines - 5} more")
                
                if value_col and value_col in viz_df.columns:
                    st.write(f"**Average {metric_type}:** {viz_df[value_col].mean():.2f}")
                    st.write(f"**Maximum {metric_type}:** {viz_df[value_col].max():.2f}")
                    st.write(f"**Minimum {metric_type}:** {viz_df[value_col].min():.2f}")
                    st.write(f"**Total {metric_type}:** {viz_df[value_col].sum():.2f}")
                
                if time_col and time_col in viz_df.columns:
                    st.write(f"**Time Range:** {viz_df[time_col].min()} to {viz_df[time_col].max()}")
                
                st.dataframe(viz_df)
            
            return True
        else:
            error_msg = f"Could not create {chart_type} chart. Missing required columns."
            logger.error(error_msg)
            st.error(error_msg)
            st.write("**Column Detection Results:**")
            st.write(f"- Time Column: {time_col}")
            st.write(f"- Machine Column: {machine_col}")
            st.write(f"- Value Column: {value_col}")
            st.write(f"- Granularity: {granularity}")
            st.write("**Available Data:**")
            st.dataframe(df.head(10))
            return False
            
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        st.write("**Debug Information:**")
        if 'df' in locals() and not df.empty:
            st.write("Data sample:")
            st.dataframe(df.head())
        return False


# Helper function to suggest optimal chart types
def suggest_chart_type(df, granularity, metric_type):
    """Suggest the best chart type based on data characteristics"""
    suggestions = []
    
    if granularity in ['hourly', 'daily']:
        suggestions.append("line - Best for showing trends over time")
        suggestions.append("bar - Good for comparing values across time periods")
    
    if granularity == 'monthly':
        suggestions.append("bar - Ideal for comparing monthly values")
        suggestions.append("line - Good for showing monthly trends")
    
    # Check if we have machine data
    machine_cols = [col for col in df.columns if 'machine' in col.lower() or 'device' in col.lower()]
    if machine_cols and df[machine_cols[0]].nunique() > 1:
        suggestions.append("pie - Useful for showing distribution across machines")
    
    return suggestions











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
                AIMessage(content="Hello! I'm your Althinect Intelligence Bot. I can help you analyze multi-machine production data with colorful visualizations! 📊\n\nTry asking: *'show the the hourly consumption of all stenters in 3rd of may 2025 using a line chart'*")
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
        query_template, date_range, query_type = detect_query_intent(user_query)

        if query_template and query_template in METABASE_QUERY_TEMPLATES:
            logger.info(f"Using predefined query template: {query_template}")
            
            start_date, end_date = date_range
            
            if query_template == 'multi_metric_daily':
                # Handle multi-metric queries
                machine_numbers = extract_machine_numbers(user_query)
                if machine_numbers:
                    # Get the first machine (or handle multiple machines later)
                    machine_key = machine_numbers[0]
                    device_names = build_multi_metric_device_names(machine_key)
                    
                    # Format the template with all device names
                    sql_query = METABASE_QUERY_TEMPLATES[query_template].format(
                        start_date=start_date,
                        end_date=end_date,
                        production_device=device_names['production_device'],
                        energy_device=device_names['energy_device'],
                        utilization_device=device_names['utilization_device']
                    )
                else:
                    # Fallback if no specific machine detected
                    sql_query = METABASE_QUERY_TEMPLATES['daily_production'].format(
                        start_date=start_date,
                        end_date=end_date,
                        device_filter=""
                    )
            else:
                # Handle single-metric queries
                device_filter = build_device_filter(user_query)
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

        
        logger.info(f"Using enhanced SQL query: {sql_query}")
    



        
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
            max_tokens=30000,
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