from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-base")
model = AutoModelForSeq2SeqLM.from_pretrained("declare-lab/flan-alpaca-base")


def generate_text(model, tokenizer, sequence, max_length):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    print(type(ids))
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=1,
        top_p=1,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))



import torch
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)
generate_text(model, tokenizer, "what is your name", 15)


async def llm_model_test():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("declare-lab/flan-alpaca-base")
    dataset = "what is your name"
    result = await freewillai.run_task(model=model, tokenizer=tokenizer, dataset=dataset)
    return result

