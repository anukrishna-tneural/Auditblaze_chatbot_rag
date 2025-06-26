from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from rag_app import SalesRAG
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re

class ChartData:
    def __init__(self, type, labels, datasets, title):
        self.type = type
        self.labels = labels
        self.datasets = datasets
        self.title = title

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  

try:
    rag_system = SalesRAG()
    logger.info("SalesRAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SalesRAG: {e}")
    rag_system = None





def check_chart_request(message):
    """Only show charts if the user explicitly asks"""
    message_lower = message.lower()
    chart_keywords = ['chart', 'graph', 'bar chart', 'pie chart', 'line chart', 'show chart', 'display chart']
    return any(keyword in message_lower for keyword in chart_keywords)

def determine_chart_type(message, query_type):
    """Determine appropriate chart type based on user request and query"""
    message_lower = message.lower()
    
    if 'pie' in message_lower:
        return 'pie'
    elif 'line' in message_lower:
        return 'line'
    else:
        return 'bar'

def try_generate_chart_from_response(response, user_message):
    """
    Try to generate chart data from RAG response
    Can return multiple charts for different categories
    """
    try:
        if not check_chart_request(user_message):
            return None
            
        lines = response.split('\n')
        charts = []  # Store multiple charts
        
        # Parse the response to identify different sections
        sections = identify_data_sections(lines)
        
        for section in sections:
            chart_data = parse_section_to_chart(section, user_message)
            if chart_data:
                charts.append(chart_data)
        
        # Return single chart or multiple charts
        if len(charts) == 1:
            return charts[0]
        elif len(charts) > 1:
            return charts  # Return list of charts
        
    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        print(f"Debug - Error generating chart data: {e}")
        print(f"Debug - Response text: {response}")
    
    return None


def identify_data_sections(lines):
    """
    Identify different data sections in the response
    Returns list of sections, each containing relevant lines
    """
    sections = []
    current_section = []
    current_title = ""
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check for section headers (titles with emojis or keywords)
        if any(keyword in line.lower() for keyword in ['💰 by revenue', '📦 by quantity', '💰 top by', '📦 top by']):
            # If we have a current section, save it
            if current_section and current_title:
                sections.append({
                    'title': current_title,
                    'lines': current_section.copy()
                })
            
            # Start new section
            current_title = line
            current_section = [line]
        
        # Add line to current section if we have one
        elif current_title and line.startswith('•'):
            current_section.append(line)
    
    # Add the last section
    if current_section and current_title:
        sections.append({
            'title': current_title,
            'lines': current_section.copy()
        })
    
    return sections


def parse_section_to_chart(section, user_message):
    """
    Parse a single section to create chart data
    Updated to handle the new response format from query_handlers.py
    """
    title = section['title']
    lines = section['lines']
    
    names = []
    values = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and section headers
        if not line or not line.startswith('•'):
            continue
        
        # Remove bullet point and parse the line
        line = line.replace('•', '').strip()
        
        # Look for patterns like "Name → AED amount, Qty quantity" or "Name → Qty quantity, AED amount"
        if '→' in line:
            parts = line.split('→')
            if len(parts) == 2:
                name = parts[0].strip()
                value_part = parts[1].strip()
                
                # Extract AED amount
                aed_match = re.search(r'AED\s+([\d,]+\.?\d*)', value_part)
                if aed_match:
                    try:
                        value_str = aed_match.group(1).replace(',', '')
                        value = float(value_str)
                        names.append(name)
                        values.append(value)
                        print(f"Debug - Added {name}: AED {value}")
                    except ValueError:
                        continue
    
    # Create chart data if we have data
    if names and values:
        # Determine chart type and styling based on content
        chart_type = determine_chart_type(user_message, title.lower())
        
        # Determine what type of data this is
        if any(keyword in title.lower() for keyword in ['salesman', 'sales person', 'sales rep', 'salesmen']):
            category = "salesman"
            colors = ['#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56', '#9966FF']
            label = "Sales Value (AED)"
        elif any(keyword in title.lower() for keyword in ['customer', 'client']):
            category = "customer" 
            colors = ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF']
            label = "Sales Value (AED)"
        elif any(keyword in title.lower() for keyword in ['product', 'item']):
            category = "product"
            colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
            label = "Sales Value (AED)"
        elif any(keyword in title.lower() for keyword in ['supplier', 'vendor']):
            category = "supplier"
            colors = ['#9966FF', '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56']
            label = "Purchase Value (AED)"
        else:
            category = "general"
            colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
            label = "Value (AED)"
        
        # Create the ChartData object
        from collections import namedtuple
        ChartData = namedtuple('ChartData', ['type', 'labels', 'datasets', 'title'])
        
        return ChartData(
            type=chart_type,
            labels=names[:10],  # Limit to top 10
            datasets=[{
                "label": label,
                "data": values[:10],
                "backgroundColor": colors[:len(names[:10])],
                "borderColor": '#fff',
                "borderWidth": 2
            }],
            title=title.replace('💰', '').replace('📦', '').strip()  # Clean up title
        )
    
    return None



@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "rag_system": "initialized" if rag_system else "failed"
    })

@app.route('/query', methods=['POST'])
def query_sales():
    if not rag_system:
        return jsonify({"error": "RAG system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        question = data['question']
        logger.info(f"Processing query: {question}")
        
        response = rag_system.query(question)
        
        return jsonify({
            "question": question,
            "answer": response,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/health", "/query", "/chat"]
    }), 404




@app.route('/chat', methods=['POST'])
def chat():
    if not rag_system:
        return jsonify({"error": "RAG system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        logger.info(f"Processing chat message: {message}")
        logger.info(f"User wants chart? {check_chart_request(message)}")
        
        # Get response from RAG system
        response = rag_system.query(message)
        
        # Prepare base response
        result = {
            "user_message": message,
            "bot_response": response,
            "timestamp": "2024-06-25T12:00:00Z",
            "status": "success"
        }
        
        # Check if user wants a chart and try to generate chart data
        if check_chart_request(message):
            chart_data = try_generate_chart_from_response(response, message)
            
            if chart_data:
                # Handle both single chart and multiple charts
                if isinstance(chart_data, list):
                    # Multiple charts
                    result["chart_data"] = []
                    for chart in chart_data:
                        result["chart_data"].append({
                            "type": chart.type,
                            "labels": chart.labels,
                            "datasets": chart.datasets,
                            "title": chart.title
                        })
                    logger.info(f"Multiple charts generated: {len(chart_data)} charts")
                else:
                    # Single chart
                    result["chart_data"] = {
                        "type": chart_data.type,
                        "labels": chart_data.labels,
                        "datasets": chart_data.datasets,
                        "title": chart_data.title
                    }
                    logger.info(f"Single chart generated with {len(chart_data.labels)} data points")
            else:
                logger.info("Chart requested but could not generate chart data from response")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500    

    

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Sales RAG API Server",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/query": "POST - Query sales data",
            "/chat": "POST - Chat interface"
        },
        "example_usage": {
            "query": {
                "method": "POST",
                "url": "/query",
                "body": {"question": "Top 5 products by revenue"}
            }
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)