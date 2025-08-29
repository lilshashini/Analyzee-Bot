# 🏭 Analyzee-Bot - Production Analytics Dashboard

This intelligent chatbot is designed to interpret natural language queries and provide real-time insights from an industrial database. It automatically converts user input into optimized SQL queries, retrieves relevant data, and generates interactive visualizations such as line charts, bar charts, and pie charts. Additionally, the chatbot includes a predictive analytics module powered by Prophet to forecast production, consumption, and utilization trends based on historical data. Built with Python, Streamlit, LangChain, MySQL, and integrated with Azure OpenAI API, the chatbot streamlines data-driven decision-making by combining conversational AI, real-time analytics, and predictive modeling in one solution.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-v0.1+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **🤖 AI-Powered Chat Interface**: Natural language queries for production data analysis
- **📊 Interactive Visualizations**: Auto-generates charts based on your questions
- **🏭 Multi-Machine Support**: Compare performance across multiple production machines
- **📈 Multiple Chart Types**: Bar, line, pie, scatter, and specialized efficiency charts
- **🔍 Smart Query Detection**: Automatically detects visualization needs and chart types
- **💾 Real-time Database Connection**: Direct MySQL database integration
- **📋 Data Export**: View and analyze raw data with expandable summaries

## 🎯 Supported Visualizations

- **Production Comparison**: Bar charts comparing output across machines
- **Efficiency Analysis**: Performance metrics visualization with color-coded efficiency scales
- **Time Series Analysis**: Line charts for trends over time
- **Pulse Rate Monitoring**: Specialized charts for machine pulse per minute data
- **Multi-Machine Dashboards**: Grouped visualizations for comprehensive analysis

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- MySQL database with production data
- Azure OpenAI API access

### 1. Clone the Repository

```bash
git clone https://github.com/lilshashini/Analyzee-Bot.git
cd Analyzee-Bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

### 4. Database Setup

Ensure your MySQL database contains production tables with the following structure:

```sql
-- Example table structure
CREATE TABLE hourly_production (
    id INT PRIMARY KEY,
    device_name VARCHAR(100),
    actual_start_time DATETIME,
    production_output DECIMAL(10,2),
    target_output DECIMAL(10,2)
);

CREATE TABLE length_data (
    id INT PRIMARY KEY,
    device_name VARCHAR(100),
    timestamp DATETIME,
    length DECIMAL(10,2)
);
```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Database Connection

1. Use the sidebar to enter your MySQL database credentials
2. Click "🔌 Connect" to establish connection
3. Verify connection status in the sidebar

### Example Queries

Try these natural language queries:

```
📊 Visualization Queries:
- "Show all three machines production by each machine in April with bar chart"
- "Plot the bar chart showing each machine's efficiency on April 1 2025"
- "Compare production by machine for last 7 days"
- "Show pulse per minute for Machine1 on June 1st"

📈 Analysis Queries:
- "Which machine had the highest efficiency in April?"
- "Show me the daily production trends"
- "Compare machine performance this month"
```

## 🔧 Configuration

### Chart Types

The bot automatically detects and creates appropriate visualizations:

- **single_bar**: Single bar charts
- **multi_machine_bar**: Grouped bar charts for multi-machine data
- **line charts**: Time series and trend analysis
- **pie charts**: Distribution and proportion charts
- **pulse_line**: Specialized pulse rate monitoring

### Database Tables

Supported table structures:
- `hourly_production`: Production data with timestamps
- `daily_production_1`: Daily aggregated production data
- `length_data`: Machine length/pulse measurements

## 🚨 Troubleshooting

### Common Issues

**Database Connection Failed**
- Verify MySQL server is running
- Check credentials in sidebar
- Ensure database exists and is accessible

**No Visualization Generated**
- Check if data exists for the specified date range
- Verify table column names match expected format
- Review SQL query in debug output

**Azure OpenAI Errors**
- Verify API key and endpoint in `.env` file
- Check deployment name matches your Azure resource
- Ensure sufficient API quota

### Debug Mode

Enable detailed logging by checking the `chatbot.log` file or Streamlit console output.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `docs/` folder for detailed guides
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join project discussions in GitHub Discussions

## 🏆 Acknowledgments

- **LangChain**: For the powerful SQL chain framework
- **Streamlit**: For the excellent web app framework
- **Plotly**: For interactive visualization capabilities
- **Azure OpenAI**: For advanced natural language processing



### 📊 Sample Queries to Get Started

```bash
# Machine Efficiency Analysis
"Show me efficiency comparison for all machines in April 2025"

# Production Trends
"give the monthly production of the machines in April using pie chart 2025"

# Multi-Machine Comparison
"Compare production output by machine for April with grouped bar chart"

# Pulse Monitoring
"Show pulse per minute variation for Machine1 1st of April 2025"


```

