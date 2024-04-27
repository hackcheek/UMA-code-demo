import pickle


model =  pickle.load(open("./sklearn_model.pickle", "rb"))

print(model.coef_.shape[1])
