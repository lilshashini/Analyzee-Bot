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
    
    # Keywords that indicate visualization request
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot',
        'bar', 'line', 'pie', 'trend', 'comparison', 'grouped bar', 'stacked bar',
        'pulse', 'pulse per minute', 'pulse rate', 'pulse variation', 'pulse trend'
    ]
    
    needs_viz = any(keyword in user_query_lower for keyword in viz_keywords)
    
    # Enhanced chart type detection with single machine support
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








def get_enhanced_sql_chain(db):
    """Enhanced SQL chain with ClickHouse-specific query generation"""
    template = """
    You are an expert data analyst. Based on the table schema below, write a ClickHouse SQL query that answers the user's question.
    
    Available Tables and Schema:
    - devices: Contains device_name, virtual_device_id
    - device_metrics: Contains device_id, timestamp, parameter, value (for length measurements)
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    IMPORTANT CLICKHOUSE GUIDELINES:
    
    1. Use ClickHouse-specific functions:
       - toTimeZone() for timezone conversion: toTimeZone(timestamp, 'Asia/Colombo')
       - toDate() for date extraction: toDate(timestamp)
       - toHour(), toMinute() for time components
       - formatDateTime() for date formatting: formatDateTime(timestamp, '%Y-%m-%d')
       
    2. For production output calculations:
       - Use parameter = 'length' from device_metrics table
       - Calculate production as difference: argMax(value, timestamp) - argMin(value, timestamp)
       - Handle counter resets with LAG functions
       - Use PARTITION BY for grouping by device and time periods
       
    3. For multi-machine production queries:
       - JOIN devices table: LEFT JOIN devices d ON device_metrics.device_id = d.virtual_device_id
       - Group by device_name and time period
       - Use toDate() for daily grouping
       - Use formatDateTime() for readable date output
    
    4. Time filtering:
       - Use timestamp >= toDateTime('2025-04-01 07:30:00', 'Asia/Colombo')
       - Use timestamp <= toDateTime('2025-04-30 19:30:00', 'Asia/Colombo')
       
    5. For pulse calculations:
       - Calculate pulse as: value - lagInFrame(value, 1) OVER (PARTITION BY device_id ORDER BY timestamp)
       - Filter WHERE parameter = 'length'
       - Use proper window functions with PARTITION BY device_id
    
    6. Column naming for visualization:
       - Use device_name AS machine_name
       - Use toDate(timestamp) AS production_date  
       - Use production_output AS daily_production
       - Use calculated efficiency AS efficiency_percent
    
    CLICKHOUSE EXAMPLES:
    
    Question: "Show all machines production by each machine in April with bar chart"
    ClickHouse Query: 
    SELECT 
        d.device_name AS machine_name,
        toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS production_date,
        argMax(dm.value, dm.timestamp) - argMin(dm.value, dm.timestamp) AS daily_production
    FROM device_metrics dm
    LEFT JOIN devices d ON dm.device_id = d.virtual_device_id
    WHERE dm.parameter = 'length'
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') >= toDateTime('2025-04-01 07:30:00', 'Asia/Colombo')
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') <= toDateTime('2025-04-30 19:30:00', 'Asia/Colombo')
    GROUP BY d.device_name, toDate(toTimeZone(dm.timestamp, 'Asia/Colombo'))
    ORDER BY production_date, machine_name
    
    Question: "Show pulse per minute for Machine1 on June 1st"
    ClickHouse Query:
    SELECT 
        d.device_name,
        toTimeZone(dm.timestamp, 'Asia/Colombo') AS timestamp,
        dm.value AS length,
        dm.value - lagInFrame(dm.value, 1) OVER (PARTITION BY dm.device_id ORDER BY dm.timestamp) AS pulse_per_minute
    FROM device_metrics dm
    LEFT JOIN devices d ON dm.device_id = d.virtual_device_id
    WHERE dm.parameter = 'length'
        AND toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) = toDate('2025-06-01')
        AND d.device_name = 'Machine1'
    ORDER BY timestamp
    
    Question: "Compare production by machine for last 7 days"
    ClickHouse Query:
    SELECT 
        d.device_name AS machine_name,
        toDate(toTimeZone(dm.timestamp, 'Asia/Colombo')) AS production_date,
        argMax(dm.value, dm.timestamp) - argMin(dm.value, dm.timestamp) AS daily_production
    FROM device_metrics dm
    LEFT JOIN devices d ON dm.device_id = d.virtual_device_id
    WHERE dm.parameter = 'length'
        AND toTimeZone(dm.timestamp, 'Asia/Colombo') >= toDateTime(subtractDays(now(), 7), 'Asia/Colombo')
    GROUP BY d.device_name, toDate(toTimeZone(dm.timestamp, 'Asia/Colombo'))
    ORDER BY production_date, machine_name
    
    Write only the ClickHouse SQL query and nothing else. Do not wrap it in backticks or other formatting.
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Keep existing Azure OpenAI configuration
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0,
        max_tokens=800,
    )
    
    def get_schema(_):
        # ClickHouse schema info - replace this with actual schema retrieval
        return """
        Tables:
        - devices: device_name (String), virtual_device_id (Int64)
        - device_metrics: device_id (Int64), timestamp (DateTime), parameter (String), value (Float64)
        
        Common parameters:
        - 'length' for production counter values
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

def get_enhanced_response(user_query: str, db, chat_history: list):
    """Enhanced response function with ClickHouse support"""
    try:
        logger.info(f"Processing user query: {user_query}")
        
        # Step 0: Check if this is a greeting or casual conversation
        if is_greeting_or_casual(user_query):
            logger.info("Detected greeting/casual conversation")
            return get_casual_response(user_query)
        
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your Althinect Intelligence Bot. I can help you analyze multi-machine production data with colorful visualizations! 📊\n\nTry asking: *'Show all three machines production by each machine in April with bar chart for all 30 day'*"),
    ]

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
    st.error("⚠️ AZURE_OPENAI_API_KEY key not found. Please add AZURE_OPENAI_API_KEY to your .env file.")
    st.stop()

st.set_page_config(page_title="Althinect Intelligence Bot", page_icon="📊")

st.title("Althinect Intelligence Bot")


with st.sidebar:
    st.subheader("⚙️ Database Settings")
    st.write("Connect to your MySQL database")











    st.text_input("Host", value="q9gou7xoo6.ap-south-1.aws.clickhouse.cloud", key="Host")
    st.text_input("Port", value="9440", key="Port")  # ClickHouse Cloud secure port
    st.text_input("User", value="shashini", key="User")
    st.text_input("Password", type="password", value="78e76q}D7£6[", key="Password")
    st.text_input("Database", value="default", key="Database")  # Update default database
    
    if st.button("🔌 Connect", type="primary"):
        with st.spinner("Connecting to ClickHouse..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("✅ Connected to ClickHouse!")
                logger.info("ClickHouse connected successfully via UI")
            except Exception as e:
                error_msg = f"❌ Connection failed: {str(e)}"
                st.error(error_msg)
                logger.error(f"ClickHouse connection failed via UI: {str(e)}")
    
    




    
    
    if "db" in st.session_state:
        st.success("🟢 Database Connected")
    
    
        with st.expander("📋 Table Information & Debugging"):
            try:
                # Test basic connection
                st.write("**Connection Test:**")
                test_result = db.query("SELECT 1")
                st.success("✅ ClickHouse connection is working")
                
                # Show current database
                try:
                    current_db_result = db.query("SELECT currentDatabase()")
                    current_db = current_db_result.result_set[0][0] if current_db_result.result_set else 'Unknown'
                    st.write(f"**Current Database:** {current_db}")
                except Exception as db_e:
                    st.write(f"**Current Database:** Could not determine - {str(db_e)}")
                
                # Show available databases
                try:
                    databases_result = db.query("SHOW DATABASES")
                    st.write("**Available Databases:**")
                    if databases_result.result_set:
                        db_list = [db[0] for db in databases_result.result_set]
                        st.write(", ".join(db_list))
                    else:
                        st.write("No databases found")
                except Exception as db_e:
                    st.write(f"**Available Databases:** Error - {str(db_e)}")
                
                # Show tables
                st.write("**Tables Discovery:**")
                
                try:
                    tables_result = db.query("SHOW TABLES")
                    if tables_result.result_set:
                        st.write("✅ SHOW TABLES result:")
                        for table in tables_result.result_set:
                            table_name = table[0] if isinstance(table, tuple) else table
                            st.write(f"  - {table_name}")
                    else:
                        st.write("❌ SHOW TABLES returned empty")
                except Exception as e1:
                    st.write(f"❌ SHOW TABLES failed: {str(e1)}")
                
                # Query system.tables
                try:
                    system_tables_result = db.query("""
                        SELECT name, database 
                        FROM system.tables 
                        WHERE database = currentDatabase()
                        ORDER BY name
                    """)
                    if system_tables_result.result_set:
                        st.write("✅ system.tables result:")
                        for table in system_tables_result.result_set:
                            st.write(f"  - {table[0]} (database: {table[1]})")
                    else:
                        st.write("❌ system.tables query returned empty")
                except Exception as e2:
                    st.write(f"❌ system.tables query failed: {str(e2)}")
                
                # Show user info
                try:
                    user_result = db.query("SELECT user()")
                    user_name = user_result.result_set[0][0] if user_result.result_set else 'Unknown'
                    st.write(f"**Connected as user:** {user_name}")
                except Exception as u_e:
                    st.write(f"**User info:** Could not determine - {str(u_e)}")
                    
            except Exception as main_e:
                st.error(f"❌ Connection test failed: {str(main_e)}")
                st.write("**Troubleshooting steps:**")
                st.write("1. Check if your credentials are correct")
                st.write("2. Verify the host and port")
                st.write("3. Ensure your user has proper permissions")
                st.write("4. Try connecting to a different database")

    else:
        st.warning("🔴 Database Not Connected")

# Chat interface
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)

user_query = st.chat_input("💬 Ask about multi-machine production data...")
if user_query is not None and user_query.strip() != "":
    logger.info(f"User query received: {user_query}")
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    with st.chat_message("assistant", avatar="🤖"):
        # Check if it's a greeting first (no database needed)
        if is_greeting_or_casual(user_query):
            response = get_casual_response(user_query)
            st.markdown(response)
        elif "db" in st.session_state:
            with st.spinner("🔄 Analyzing data and creating visualization..."):
                response = get_enhanced_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
        else:
            response = "⚠️ Please connect to the database first using the sidebar to analyze production data."
            st.markdown(response)
            logger.warning("User attempted to query without database connection")
        
    st.session_state.chat_history.append(AIMessage(content=response))
    logger.info("Conversation turn completed")
