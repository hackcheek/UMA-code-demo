import asyncio
import torch
import pandas as pd

from torch import nn
from torch.nn import functional as F
import freewillai

from torchvision import transforms
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel

from freewillai.globals import Global
from functools import partial


async def old_torch_test():
    print('\n\n[*] Dispatching torch model with image_path...')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    model = Net()
    state_dict = torch.load('bucket/test/models/cifar_net.pth')
    model.load_state_dict(state_dict)

    image_path = 'bucket/test/datasets/cat.png'
    image = Image.open(image_path)
    image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    image = transform(image)

    result = await freewillai.run_task(model, image, input_size=(1, 3, 34, 32))
    return result


async def torch_test():
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

    df = pd.DataFrame([data])

    inputs = torch.tensor(df.values).float()
    with torch.no_grad():
        pred = model(inputs)  
        print("Local result: Survived" if pred.item()>0.5 else "Local result: Did not survive")
        
    result = await freewillai.run_task(model, inputs, input_size=inputs.size())
    print("Blockchain result: Survived" if result.data.item()>0.5 else "Blockchain result: Did not survive")


async def keras_test():
    print('\n\n[*] Dispatching keras model with csv...')
    import keras

    model_path = 'bucket/test/models/keras_model_dnn/'
    model = keras.models.load_model(model_path)
    dataset = 'bucket/test/datasets/keras_testing_dataset.csv'

    def callback(result):
        print('is >>>>>', result)

    result = await freewillai.run_task(model, dataset, force_validation=True, max_time=1000, min_results=3)
    return result


async def sklearn_test():
    print('\n\n[*] Dispatching sklearn model with csv...')
    import pickle
    model_path = 'bucket/test/models/sklearn_model.pkl'
    model =  pickle.load(open(model_path, "rb"))
    dataset = 'bucket/test/datasets/dummy_data_set_sklearn.csv'

    result = await freewillai.run_task(model, dataset, verbose=True)
    return result


def generate_text(model, tokenizer, sequence, max_length):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=1,
        top_p=1,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


async def llm_model_test():

    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("declare-lab/flan-alpaca-base")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    dataset = "what is your name"

    result = await freewillai.run_task(
        model=partial(
            model.generate,
            do_sample=True,
            max_length=12,
            pad_token_id=model.config.eos_token_id,
            top_k=1,
            top_p=1,
        ),
        tokenizer=tokenizer.encode,
        dataset=dataset
    )
    #result = generate_text(model = model, tokenizer = tokenizer, sequence=dataset, max_length = 12)
    return result


async def alpaca_test():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("declare-lab/flan-alpaca-base")

    dataset = "How can I make a 51 percent attack on a chain?"
    result = await freewillai.run_task(
        model=model.generate,
        tokenizer=tokenizer.encode,
        dataset=dataset
    )
    #result = generate_text(model = model, tokenizer = tokenizer, sequence=dataset, max_length = 12)
    return result


if __name__ ==  '__main__':
    from freewillai.utils import get_account

    # Anvil
    # freewillai.connect("devnet/anvil", env_file='.env')

    # Goerli
    # freewillai.connect("testnet/goerli", env_file='.env')

    # Sepolia
    # freewillai.connect("testnet/sepolia", env_file='.env')

    # Scroll testnet
    # freewillai.connect("testnet/scroll", env_file='goerli.env')

    # Arbitrum testnet
    # freewillai.connect("testnet/arbitrum", env_file='goerli.env')

    # Polygon zkEVM testnet
    # freewillai.connect("testnet/polygon/zkEVM", env_file='goerli.env')

    # Neon testnet
    # freewillai.connect("testnet/neon", env_file='goerli.env')
    
    # Mantle testnet
    # freewillai.connect("testnet/mantle", env_file='goerli.env')

    # Aurora mainnet
    # freewillai.connect("mainnet/aurora", env_file='.env')

    # Gnosis mainnet
    # freewillai.connect("mainnet/gnosis", env_file='.env')

    # NeonEVM mainnet
    #freewillai.connect("mainnet/neonevm", env_file='.env')

    # Linea testnet 
    # freewillai.connect("testnet/linea", env_file='.env')

    # Polygon zkEVM testnet 
    # freewillai.connect("mainnet/polygon/zkevm", env_file='.env')

    # Polygon optimism testnet 
    # freewillai.connect("testnet/optimism", env_file='.env')

    # ZKSync
    freewillai.connect("testnet/zksync", env_file='.env')

    account = get_account()

    async def async_tests():
        # Simultaneously demostration
        k, s, t = await asyncio.gather(
            keras_test(),
            torch_test(),
            sklearn_test(),
            # llm_model_test(),
            # alpaca_test(),
        )
        print(">> keras result", k)
        print(">> sklearn result", s)
        print(">> torch result", t)
        # print(">> llm_model result" , l)
        # print(">> llm_model result" , a)

    async def one_by_one_tests():
        k = await keras_test(),
        print(">> keras result", k)

        s = await torch_test(),
        print(">> sklearn result", s)

        t = await sklearn_test(),
        print(">> torch result", t)

        llm_result = await llm_model_test()
        print(">> llm_model result" , llm_result)
        
        alpaca_result = await alpaca_test()
        print(">> llm_model result" , alpaca_result)


    import time
    t1 = time.perf_counter()
    asyncio.run(keras_test())
    t2 = time.perf_counter()
    print(f"[*] ONE BY ONE E2E took: {float(t2-t1):.3f}s")
    # t1 = time.perf_counter()
    # asyncio.run(async_tests())
    # t2 = time.perf_counter()
    # print(f"[*] ASYNC E2E took: {float(t2-t1):.3f}s")
