import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
import evaluate
from classifier_model import PreEmbeddedSequenceClassification
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer



unique_labels = ["Crisis finance", "PAF"]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}


def remap_binary(example):
    example['label'] = 'PAF' if 'PAF' in example['labels'] else 'Crisis finance'
    example['label'] = label2id[example['label']]
    return example


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 30000
eval_interval = 300
learning_rate = 5e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------
out_dir = "out-gist"

dv = 'embedding'
shuffle = False
stratify_column = None

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_keys.append('stratify_column')
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging


os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(predictions, labels):
    return clf_metrics.compute(predictions=predictions, references=labels)


def main():
    # dataset = load_dataset("alex-miller/cdp-paf-meta-limited", split="train")
    # dataset = dataset.filter(lambda example: example['labels'] != 'Unrelated')
    # dataset = dataset.map(remap_binary, num_proc=8, remove_columns=["labels"])
    # # Rebalance
    # positive = dataset.filter(lambda example: example["label"] == 1)
    # negative = dataset.filter(lambda example: example["label"] == 0)
    # negative = negative.shuffle(seed=42)
    # negative = negative.select(range(positive.num_rows))
    # dataset = concatenate_datasets([negative, positive])
    # # Embed
    # embedding_model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
    # embeddings = embedding_model.encode(dataset['text'], convert_to_tensor=True)
    # dataset = dataset.add_column('embedding', embeddings.tolist())
    # # Split
    # dataset = dataset.class_encode_column('label').train_test_split(
    #     test_size=0.2,
    #     stratify_by_column="label",
    #     shuffle=True,
    #     seed=42
    # )
    # dataset.save_to_disk("paf-pre-embedded-gist")
    dataset = load_from_disk("paf-pre-embedded-gist")
    train_x = [torch.tensor(emb, dtype=torch.float32) for emb in dataset['train'][config['dv']]]
    test_x = [torch.tensor(emb, dtype=torch.float32) for emb in dataset['test'][config['dv']]]
    train_y = [int(l) for l in dataset['train']['label']]
    test_y = [int(l) for l in dataset['test']['label']]
    print("Training stratification: {}%".format(
        round((sum(train_y)/len(train_y))*100, 1)
    ))
    print("Testing stratification: {}%".format(
        round((sum(test_y)/len(test_y))*100, 1)
    ))

    model_args = {
        "num_labels": len(id2label.keys()),
        "dim": 384,
        "seq_classif_dropout": 0.6
    }
    model = PreEmbeddedSequenceClassification(model_args)

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data_x = train_x if split == 'train' else test_x
        data_y = train_y if split == 'train' else test_y
        ix = torch.randint(len(data_x), (batch_size,))
        x = torch.stack([data_x[i] for i in ix])
        y = torch.tensor([data_y[i] for i in ix], dtype=int)
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            accuracies = torch.zeros(eval_iters)
            precisions = torch.zeros(eval_iters)
            recalls = torch.zeros(eval_iters)
            f1s = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                if split == 'val':
                    pred = logits.max(1).indices
                    metrics = compute_metrics(pred, Y)
                    accuracies[k] = metrics['accuracy']
                    precisions[k] = metrics['precision']
                    recalls[k] = metrics['recall']
                    f1s[k] = metrics['f1']
                losses[k] = loss.item()
            out[split] = losses.mean()
            out['accuracy'] = accuracies.mean()
            out['precision'] = precisions.mean()
            out['recall'] = recalls.mean()
            out['f1'] = f1s.mean()
        model.train()
        return out
    
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = 1e9
    for iter_num in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val acc {losses['accuracy']:.4f}, val precision {losses['precision']:.4f}, val recall {losses['recall']:.4f}, val f1 {losses['f1']:.4f}"
            )
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'model_args': model_args,
                        'optimizer': optimizer.state_dict(),
                        'iter': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
