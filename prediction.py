from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)

# Database Configuration
DB_URI = os.getenv("DB_URI")
engine = create_engine(DB_URI)

# Time Intervals Mapping
TIME_INTERVALS = {
    "1h": "1H",
    "4h": "4H",
    "8h": "8H",
    "12h": "12H",
    "1d": "D",
    "1w": "W",
    "1m": "ME",  # Month End
    "1y": "ME"   # Month End for yearly aggregation
}

def get_order_data(shopname):
    """Fetches raw order data without resampling."""
    query = """
        SELECT created_at, COUNT(*) AS orders
        FROM orders
        WHERE shopname = %s
        GROUP BY created_at
        ORDER BY created_at;
    """
    try:
        df = pd.read_sql(query, engine, params=(shopname,))
        if df.empty:
            return None

        df["created_at"] = pd.to_datetime(df["created_at"])
        df.set_index("created_at", inplace=True)
        
        return df
    except Exception as e:
        print(f"Database query error: {e}")
        return None

def predict_orders(shopname, time_range="1d"):
    """Unified prediction approach for all time ranges."""
    # Get raw order data
    df = get_order_data(shopname)
    
    if df is None or df.empty or len(df) < 5:
        return {"shopname": shopname, "predicted_orders": "Not enough data"}
    
    try:
        # Start with daily resampling for consistency
        daily_df = df.resample('D').sum().fillna(0)
        
        # Calculate average daily orders
        daily_avg = daily_df['orders'].mean()
        
        # Get the total number of days in the dataset
        total_days = (daily_df.index.max() - daily_df.index.min()).days + 1
        
        # Print debug info
        print(f"Total days in data: {total_days}")
        print(f"Average daily orders: {daily_avg}")
        print(f"Total orders: {daily_df['orders'].sum()}")
        
        # Set prediction based on time range
        if time_range == "1h":
            # Hourly - fraction of daily average
            prediction = daily_avg / 24
            future_date = df.index[-1] + pd.DateOffset(hours=1)
            
        elif time_range == "4h":
            prediction = daily_avg / 6  # 4 hours is 1/6 of a day
            future_date = df.index[-1] + pd.DateOffset(hours=4)
            
        elif time_range == "8h":
            prediction = daily_avg / 3  # 8 hours is 1/3 of a day
            future_date = df.index[-1] + pd.DateOffset(hours=8)
            
        elif time_range == "12h":
            prediction = daily_avg / 2  # 12 hours is 1/2 of a day
            future_date = df.index[-1] + pd.DateOffset(hours=12)
            
        elif time_range == "1d":
            prediction = daily_avg
            future_date = df.index[-1] + pd.DateOffset(days=1)
            
        elif time_range == "1w":
            prediction = daily_avg * 7  # 7 days in a week
            future_date = df.index[-1] + pd.DateOffset(weeks=1)
            
        elif time_range == "1m":
            prediction = daily_avg * 30  # Approximate 30 days in a month
            future_date = df.index[-1] + pd.DateOffset(months=1)
            
        elif time_range == "1y":
            prediction = daily_avg * 365  # 365 days in a year
            future_date = df.index[-1] + pd.DateOffset(years=1)
        
        # Try to refine predictions if we have enough data
        if total_days >= 60 and time_range in ["1m", "1y"]:
            try:
                # Monthly aggregation
                monthly_df = df.resample('ME').sum()
                if len(monthly_df) >= 6:  # Need at least 6 months of data
                    # Use Exponential Smoothing for better forecasting
                    model = ExponentialSmoothing(
                        monthly_df["orders"].values,
                        trend='add',
                        seasonal='add' if len(monthly_df) >= 12 else None,
                        seasonal_periods=12 if len(monthly_df) >= 12 else None
                    )
                    model_fit = model.fit()
                    
                    if time_range == "1m":
                        monthly_prediction = model_fit.forecast(1)[0]
                        prediction = max(monthly_prediction, daily_avg * 30)
                    elif time_range == "1y":
                        yearly_prediction = sum(model_fit.forecast(12))
                        prediction = max(yearly_prediction, daily_avg * 365)
                    
                    method = "Exponential Smoothing"
                else:
                    method = "Daily Average × Period Length"
            except Exception as e:
                print(f"Advanced forecasting error: {e}")
                method = "Daily Average × Period Length"
        else:
            method = "Daily Average × Period Length"
        
        # Ensure predictions are consistent with time periods
        if time_range == "1y" and prediction < daily_avg * 300:
            # Yearly prediction should be at least 300 days worth
            prediction = daily_avg * 365
            method = "Adjusted Daily Average × 365"
            
        if time_range == "1m" and prediction > daily_avg * 40:
            # Monthly prediction shouldn't be more than ~40 days worth
            prediction = daily_avg * 30
            method = "Adjusted Daily Average × 30"
            
        # Ensure yearly prediction is greater than monthly
        if time_range == "1y" and prediction <= daily_avg * 30:
            prediction = daily_avg * 365
            method = "Enforced Yearly Minimum"

        # Make sure predictions are not too low
        prediction = max(prediction, 1)
        
        return {
            "shopname": shopname,
            "predicted_orders": int(round(prediction)),
            "date": future_date.strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "avg_daily_orders": round(daily_avg, 2)
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"shopname": shopname, "predicted_orders": f"Error: {str(e)}"}

@app.route("/predict_orders", methods=["GET"])
def get_predictions():
    """API endpoint to fetch order predictions."""
    shopname = request.args.get("shopname")
    time_range = request.args.get("timeRange", "1d")

    if not shopname:
        return jsonify({"error": "Shopname parameter is required"}), 400
    if time_range not in TIME_INTERVALS:
        return jsonify({"error": "Invalid time range"}), 400

    result = predict_orders(shopname, time_range)
    return jsonify(result)

#student chatbot
SYSTEM_PROMPT = """
You are a helpful print shop assistant for campus students. You can help students with:
1. Information about their past print orders
2. How to place new print orders
3. Print service options (color, black/white, binding, paper sizes)
4. Answering questions about delivery times and pricing
5. Status of their current orders

### **How to Place an Order:**  
1. **Enter Shops:** You can only place orders at shops that are currently open.  
2. **Select Print Requirements:** Choose options like color or black & white, paper size, binding, and number of copies.  
3. **Upload Documents:** Upload your files securely.  
4. **Pay the Bill:** Complete the payment through the available payment methods.  
5. **Order Confirmation:** You will receive an **email confirmation** when your order is placed successfully.  
6. **Order Collection:** Once your prints are ready, you will receive another **email notification**. You can then visit the respective shop to collect your order.  

### **Additional Information:**  
- **Who to Contact for Issues?** If you face any issues with your order, you should directly contact the respective shop where you placed the order. If the order is falied contact the respective shop  
- **Major Issues:** For urgent concerns or unresolved issues, contact **rr9589@srmist.edu.in**.  
- **Viewing Past Orders:** You can check your past print orders, including details like dates, number of copies, document names, and costs.  
- **Delivery & Processing Time:** Printing is usually completed within an hour. However, during peak times, there might be slight delays.  

Be concise, friendly, and helpful! Remember, this is a **campus print shop service**, so your guidance should be specific to student printing needs.

"""

def get_student_order_history(username):
    """Fetch print order history for a specific student from the database."""
    try:
        query = """
        SELECT id, shopname, copies, total, created_at, status, payment_id
        FROM orders
        WHERE username = :username
        ORDER BY created_at DESC
        """
        
        with engine.connect() as connection:
            result = connection.execute(text(query), {"username": username})
            
            # Try different methods to convert to dictionary
            try:
                # Modern SQLAlchemy method
                orders = [dict(row) for row in result.mappings()]
            except AttributeError:
                try:
                    # Alternative method
                    orders = [row._asdict() for row in result]
                except AttributeError:
                    # Fallback method
                    orders = []
                    column_names = result.keys()
                    for row in result:
                        orders.append({column: value for column, value in zip(column_names, row)})
            
        # Rest of your function remains the same
        if not orders:
            return "No print order history found for this student."
            
        # Convert to a readable format
        formatted_orders = []
        for i, order in enumerate(orders):
            order_str = f"Order #{i+1} - {order['shopname']} - {order['created_at'].strftime('%Y-%m-%d %H:%M')}\n"
            order_str += f"Copies: {order['copies']} | Total: {order['total']} Rs | Status: {order['status']}"
            if order['payment_id']:
                order_str += f" | Payment ID: {order['payment_id']}"
            formatted_orders.append(order_str)
        
        return "\n\n".join(formatted_orders)
        
    except Exception as e:
        print(f"Error fetching order history: {e}")
        return f"Error fetching order history: {str(e)}"

def generate_student_chatbot_response(input_text, username,conversation_history):
    """Generate a response using Google Gemini API with student-specific context."""
    try:
        # Get student's order history
        order_history = get_student_order_history(username)
        
        # Build context for the model
        context = f"""
        Student username: {username}
        Order History: {order_history}
        Conversation History: {conversation_history}

        Answer the user quries polietly and more precise.
        Don't ever talk about something which is not related.
        Don't talk about errors.
        Don't use * or # because those doesnt work.
        """
        
        # Configure the model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )
        
        # Prepare the prompt
        prompt = f"{SYSTEM_PROMPT}\n\nContext Information:\n{context}\n\nUser: {input_text}\n\nAssistant:"
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        print(f"Error generating chatbot response: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."

@app.route('/chatbotstudent', methods=['POST'])
def student_chatbot():
    """API endpoint for the chatbot."""
    data = request.json
    user_message = data.get('message', '')
    username = data.get('username', '')
    conversation_history = data.get("conversation_history", [])
    if not user_message or not username:
        return jsonify({'error': 'Message and username are required'}), 400
    
    formatted_history = "\n".join(
        [f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation_history]
    )
    response = generate_student_chatbot_response(user_message, username,formatted_history)
    
    return jsonify({'response': response})

# shopkeeper chatbot
SYSTEM_PROMPT = """
You are a business intelligence assistant for shopkeepers. You provide:
1. Insights on their past and current print orders.
2. Strategies to improve earnings and business efficiency.
3. Predictions on future demand and sales trends.
4. Guidance on completing or failing an order using the toggle option in the dashboard.
5. Explanation of the billing system and monthly payouts.

### **How to Manage Orders:**  
1. **Order Status Update:** Each order has a toggle option in the dashboard:
   - Switching to **Completed** marks the order as finished.
   - Switching to **Failed** requires the shopkeeper to either refund the student or create a new order.
2. **Handling Failed Orders:** If an order fails, shopkeepers must:
   - Either **refund** the amount to the student.
   - Or **reprocess** the order manually when the student arrives at the shop.

### **Billing & Payments:**  
- All payments made via the app are collected into a **hot wallet**.
- Monthly, earnings are distributed to shopkeepers based on their revenue.
- This includes payments for failed orders, as shopkeepers are expected to either refund or reprocess these orders.

### **Additional Information:**  
- **Major Issues:** For unresolved problems, contact **rr9589@srmist.edu.in**.
- **Business Growth:** You can ask for strategies to enhance your print shop revenue and manage demand efficiently.

Be precise, professional, and helpful! Your focus is on assisting shopkeepers in managing orders and growing their business.
"""

def get_shopkeeper_order_history(username):
    """Fetch order history for a shopkeeper from the database."""
    try:
        query = """
        SELECT * FROM orders
        WHERE shopname = :username
        ORDER BY created_at DESC
        """
        
        with engine.connect() as connection:
            result = connection.execute(text(query), {"username": username})
            
            try:
                orders = [dict(row) for row in result.mappings()]
            except AttributeError:
                try:
                    orders = [row._asdict() for row in result]
                except AttributeError:
                    orders = []
                    column_names = result.keys()
                    for row in result:
                        orders.append({column: value for column, value in zip(column_names, row)})
        
        if not orders:
            return "No print order history found for this shopkeeper."
        
        formatted_orders = []
        for i, order in enumerate(orders):
            order_str = f"Order #{i+1} - {order['username']} - {order['created_at'].strftime('%Y-%m-%d %H:%M')}\n"
            order_str += f"Copies: {order['copies']} | Total: {order['total']} Rs | Status: {order['status']}"
            if order['payment_id']:
                order_str += f" | Payment ID: {order['payment_id']}"
            formatted_orders.append(order_str)
        
        return "\n\n".join(formatted_orders)
        
    except Exception as e:
        print(f"Error fetching order history: {e}")
        return f"Error fetching order history: {str(e)}"

def generate_shopkeeper_chatbot_response(input_text, username, conversation_history):
    """Generate a response using Google Gemini API with shopkeeper-specific context."""
    try:
        order_history = get_shopkeeper_order_history(username)
        
        context = f"""
        Shopkeeper username: {username}
        Order History: {order_history}
        Conversation History: {conversation_history}

        Answer the user queries politely and concisely.
        Keep responses business-focused.
        Do not discuss errors or unrelated topics.
        Don't use ** or # because those doesnt work ,that means don't use bold words.
        """
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )
        
        prompt = f"{SYSTEM_PROMPT}\n\nContext Information:\n{context}\n\nUser: {input_text}\n\nAssistant:"
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        print(f"Error generating chatbot response: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."

@app.route('/chatbotshopkeeper', methods=['POST'])
def shopkeeper_chatbot():
    """API endpoint for the shopkeeper chatbot."""
    data = request.json
    user_message = data.get('message', '')
    username = data.get('username', '')
    conversation_history = data.get("conversation_history", [])
    
    if not user_message or not username:
        return jsonify({'error': 'Message and username are required'}), 400
    
    formatted_history = "\n".join(
        [f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation_history]
    )
    response = generate_shopkeeper_chatbot_response(user_message, username, formatted_history)
    
    return jsonify({'response': response})
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, port=port)