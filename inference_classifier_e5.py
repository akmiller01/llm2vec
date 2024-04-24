import os
from datasets import Dataset
from classifier_model import PreEmbeddedSequenceClassification
import torch
from sentence_transformers import SentenceTransformer


unique_labels = ["Crisis finance", "PAF"]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 30000
eval_interval = 300
learning_rate = 1e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------
out_dir = "out-e5"


config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging


os.makedirs(out_dir, exist_ok=True)


def main():

    # model init
    print(f"Resuming from {out_dir}")
    # resume from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]
    model = PreEmbeddedSequenceClassification(model_args)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter"]
    best_val_loss = checkpoint["best_val_loss"]
    model.to(device)

    config = checkpoint["config"]

    text_data = {'text': [
        'Climate Risk Adaptation and Insurance in the Caribbean (CRAIC) CLIMATE RISK ADAPTATION AND INSURANCE IN THE CARIBBEAN (CRAIC) As an integral part of climate risk management, climate risk insurance is becoming increasingly important for securing livelihoods, especially in countries that are increasingly exposed to extreme weather events. The project is therefore intensifying the dissemination of these insurance policies in selected Caribbean countries, extending the availability of the Livelihood Protection Policy to other countries, and helping to increase the number of insurance providers. The target groups are mainly sections of the population that are strongly affected by climate change. The project is also integrating climate risk insurance more strongly into the national disaster management policy. The implementing partner is the Caribbean Catastrophe Risk Insurance Facility (CCRIF), an insurance pool set up by the Caribean Community (CARICOM) states to insure themselves against natural disasters. The project is also participating in regional & international climate policy processes and negotiation.',
        'Humanitarian Assistance - Improving Emergency Preparedness HUMANITARIAN ASSISTANCE - IMPROVING EMERGENCY PREPAREDNESS Improving forecast-based Emergency Preparedness for climate risks in Bangladesh, Philippines, Nepal, Haiti, Dominican Republic Improving Emergency Preparedness Verbesserung Katastrophennothilfe Improving forecast-based Emergency Preparedness for climate risks in Bangladesh, Philippines, Nepal, Haiti, Dominican Republic Verbesserung von vorhersagenbasierter Katastrophennothilfe für Klimarisiken in Bangladesh, Philippinen, Nepal, Haiti und der Dominikanischen Republik.',
        'Increase the resilience of livelihoods to threats and crises INCREASE THE RESILIENCE OF LIVELIHOODS TO THREATS AND CRISES Increase the resilience of livelihoods to threats and crises'
    ]}
    text_data = Dataset.from_dict(text_data)
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    embeddings = embedding_model.encode(text_data['text'], convert_to_tensor=True, normalize_embeddings=True)
    text_data = text_data.add_column('embedding', embeddings.tolist())
    x = torch.stack([torch.tensor(emb, dtype=torch.float32) for emb in text_data[config['dv']]])

    with torch.no_grad():
        torch.manual_seed(1337)
        logits, loss = model(x)
    pred = logits.max(1).indices
    pred = pred.tolist()
    probs = torch.nn.functional.softmax(logits, dim=1)[:,1]
    probs = probs.tolist()
    
    for i, text in enumerate(text_data['text']):
        print(text)
        print(id2label[pred[i]])
        print("{}%".format(round(probs[i] * 100, 2)))
        print("\n")


if __name__ == '__main__':
    main()
