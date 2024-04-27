import freewillai
import pickle
import numpy as np

filename = 'demos/python/sklearn/model_prediction_houses_sklearn2.pkl'
model = pickle.load(open(filename, 'rb'))

data = {
    "bedrooms": 3, 
    "bathrooms": 1,
    "sqft_living": 200,
    "sqft_lot": 3362,
    "floors": 5,
    "waterfront": 1,   # 0 or 1
    "view": 0,         # 0 or 1
    "condition": 1,    # 5 is best, 1 is worst
    "sqft_above": 1050,
    "sqft_basement": 90,
    "yr_built": 2023
}

data = np.array([list(data.values())])

# Predict house price locally:
local_result = model.predict(data)

# Predict house price on the blockchain:
result = await freewillai.run_task(model, data)

print("Result validated by freewillai nodes:", result)
