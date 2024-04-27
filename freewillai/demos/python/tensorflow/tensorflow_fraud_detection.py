# Fraud detection
# This program detects fraud based on transaction details

from tensorflow.keras.models import load_model
import numpy as np

model = load_model('demos/python/tensorflow/fraud_detection_model.h5')

data = {
    "amount" : 0.15,  # Transaction amount, any positive value is possible
    "is_weekend":  0,  # can be 0 (not weekend) or 1 (weekend)
    "is_night" : 0,  # can be 0 (not night) or 1 (night)
    "customer_num_transactions_1_day" : 4,  # Number of transactions by this customer in last 24 hours, any non-negative integer is possible
    "customer_num_transactions_7_day" : 22,  # Number of transactions by this customer in last 7 days, any non-negative integer is possible
    "customer_num_transactions_30_day" : 81,  # Number of transactions by this customer in last 30 days, any non-negative integer is possible
    "customer_avg_amount_1_day" : 37.7,  # Average value of transactions by this customer in last 24 hours, any non-negative value is possible
    "customer_avg_amount_7_day" : 14.28,  # Average value of transactions by this customer in last 7 days, any non-negative value is possible
    "customer_avg_amount_30_day" : 9.3,  # Average value of transactions by this customer in last 30 days, any non-negative value is possible
    "terminal_num_transactions_1_day" : 1,  # Number of transactions made through this terminal in last 24 hours, any non-negative integer is possible
    "terminal_num_transactions_7_day" : 9,  # Number of transactions made through this terminal in last 7 days, any non-negative integer is possible
    "terminal_num_transactions_30_day" : 22,  # Number of transactions made through this terminal in last 30 days, any non-negative integer is possible
    "terminal_fraud_risk_1_day" : 0,  # Fraud risk for this terminal for transactions made in last 24 hours, calculated with a 7-day delay. Should be between 0 and 1 (inclusive).
    "terminal_fraud_risk_7_day" : 0,  # Fraud risk for this terminal for transactions made in last 7 days, calculated with a 7-day delay. Should be between 0 and 1 (inclusive).
    "terminal_fraud_risk_30_day" : 0  # Fraud risk for this terminal for transactions made in last 30 days, calculated with a 7-day delay. Should be between 0 and 1 (inclusive).
}

data = np.array([list(data.values())])
# predict fraud locally
prediction = model.predict(data)
binary_prediction = "Fraud" if prediction > 0.5 else "Legitimate"

# predict fraud on the blockchain
result = await freewillai.run_task(model, data)

print("Local prediction: ", prediction, binary_prediction)
print("Blockchain result: ", result.data)
