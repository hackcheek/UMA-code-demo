import torch
import torch.nn as nn
import pandas as pd

class TitanicSurvivalNN(nn.Module):
    def __init__(self):
        super(TitanicSurvivalNN, self).__init__()
        self.fc1 = nn.Linear(10, 16) 
        self.fc2 = nn.Linear(16, 32) 
        self.fc3 = nn.Linear(32, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  
        return x

model = TitanicSurvivalNN()

model.load_state_dict(torch.load('demos/python/pytorch/model_titanic.pth'))    # loading the weights

model.eval()

data = {
    "Pclass": 1,  # 0 - False, 1 - True
    "Sex": 1,     # 0 - female, 1 - male 
    "Age": 82.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 71.2833,
    "Cabin": 7,          # 0-8 representing different decks and cabin locations 0 - Deck A, 1 - Deck B...
    "Embarked": 5,       # Embarkation: 0 - Cherbourg, 1 - Queenstown, 2 - Southampton
    "Family onboard": 1,
    "Title": 2           #Title: 0 - Master, 1 - Mr, 2 - Mrs, 3 - Notable
}

inputs = torch.tensor(list(data.values())).float()

with torch.no_grad():
    local_result = model(inputs)  
    
fwai_result = await freewillai.run_task(model, inputs, input_size=inputs.size())

print("Local result: Survived" if local_result.item()>0.5 else "Local result: Did not survive")
print("Blockchain result: Survived" if fwai_result.data.item()>0.5 else "Blockchain result: Did not survive")
