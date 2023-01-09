"""
Эксперимент с восстановлением текста из одного вектора эмбеддинга, помещаемого вместо вектора первого токена
в авторегрессионной модели GPT.
"""

import os
import random

import numpy as np
import tqdm
import sklearn.model_selection
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
import sentence_transformers


class FinetuneDataset(Dataset):
    def __init__(self, samples, gpt_model):
        self.gpt_embedding = gpt_model.base_model.wte
        self.max_len = 0
        self.samples = list(samples)
        self.max_len = max(len(sample['tokens']) for sample in samples)
        self.pad_token_id = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        npad = self.max_len - len(sample['tokens'])
        input_ids = sample['tokens'] + npad*[self.pad_token_id]
        labels = [-100] + sample['tokens'] + npad*[-100]

        with torch.no_grad():
            v1 = torch.FloatTensor([sample['embedding']]).to(device)
            v2 = self.gpt_embedding(torch.LongTensor(input_ids).to(device))
            input_vectors = torch.vstack((v1, v2))

        return input_vectors, torch.LongTensor(labels).to(device)


def train(model, train_batch_generator, optimizer, eval_steps, eval_batch_generator, viz_samples):
    total_loss = 0
    for istep, (input_vectors, labels) in tqdm.tqdm(enumerate(train_batch_generator, start=1), desc='Training', total=len(train_batch_generator)):
        model.train()
        outputs = model.forward(inputs_embeds=input_vectors,labels=labels, attention_mask=None)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if 0 == (istep % eval_steps):
            print('\n\nVisualization:')
            for sample in random.choices(population=viz_samples, k=5):
                visualize(gpt_model, sample)
            eval_loss = test(model, eval_batch_generator)
            print('\nStep: {} Eval loss: {}\n'.format(istep, eval_loss))

    avg_train_loss = total_loss / len(train_batch_generator)
    return avg_train_loss


def test(model, batch_generator):
    model.eval()
    total_loss = 0
    for input_vectors, labels in batch_generator:
        model.eval()
        outputs = model.forward(inputs_embeds=input_vectors, labels=labels, attention_mask=None)
        loss = outputs.loss
        total_loss += loss.item()

    avg_train_loss = total_loss / len(batch_generator)
    return avg_train_loss



def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def visualize(model, sample):
    temperature = 1.0
    top_p = 0.85
    top_k = 50

    current_output_ids = []
    max_len = 100
    v1 = torch.FloatTensor([sample['embedding']]).to(device)
    while len(current_output_ids) < max_len:
        with torch.no_grad():
            v2 = gpt_model.base_model.wte(torch.LongTensor(current_output_ids).to(device))
            input_vectors = torch.vstack((v1, v2)).unsqueeze(dim=0)
            o = model(inputs_embeds=input_vectors)

        logits = o.logits
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)

        #prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        prev = torch.multinomial(probs, 1)
        if prev.item() == gpt_tokenizer.eos_token_id:
            break
        current_output_ids.append(prev.item())

    output_text = gpt_tokenizer.decode(current_output_ids)
    print('{} ==> {}'.format(sample['text'], output_text))


proj_dir = os.path.expanduser('~/polygon/chatbot')

# Загружаем список предложений. Вообще чем их больше и чем они разнообразнее, тем лучше, поэтому
# имеет смысл дополнять этот набор своими текстами.
texts = set()
max_text_len = 60
with open('texts.txt', 'r', encoding='utf-8') as rdr:
    for line in rdr:
        s = line.strip()
        if max_text_len > len(s) > 1:
            texts.add(s)

texts = list(texts)
print('{} texts'.format(len(texts)))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('device={}'.format(device))

gpt_model_name = 'sberbank-ai/rugpt3large_based_on_gpt2'
gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
gpt_tokenizer.add_special_tokens({'eos_token': '</s>', 'pad_token': '<pad>'})

gpt_model = transformers.AutoModelForCausalLM.from_pretrained(gpt_model_name)
gpt_model.to(device)

gpt_embed_dim = gpt_model.transformer.embed_dim

samples = []

# Эмбеддер для получения вектора из текста.
embedder_model_name = 'sentence-transformers/LaBSE'
print('Loading embedder model "{}"...'.format(embedder_model_name))
embedder = sentence_transformers.SentenceTransformer(embedder_model_name, device="cuda" if use_cuda else "cpu")

with tqdm.tqdm('Vectorization', total=len(texts)) as pbar:
    texts_ = list(texts)
    batch_size = 256
    while len(texts_) > 0:
        batch = texts_[:batch_size]
        texts_ = texts_[batch_size:]
        embeddings = embedder.encode(batch)
        vx = embeddings.tolist()
        for text, v in zip(batch, vx):
            tokens = gpt_tokenizer.encode(text+'</s>')
            if len(v) < gpt_embed_dim:
                v = v + [0.0] * (gpt_embed_dim - len(v))
            samples.append({'text': text, 'tokens': tokens, 'embedding': v})

        pbar.update(len(batch))
del embedder

train_samples, test_samples = sklearn.model_selection.train_test_split(samples, test_size=0.05)

train_dataset = FinetuneDataset(train_samples, gpt_model)
test_dataset = FinetuneDataset(test_samples, gpt_model)

optimizer = optim.AdamW(gpt_model.parameters(), lr=1e-5)

batch_size = 16
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1)

epochs = 2

best_loss = np.inf
for epoch in range(1, epochs + 1):
    print('\n=== EPOCH {}/{} ==='.format(epoch, epochs))
    try:
        train_loss = train(gpt_model, train_generator, optimizer, eval_steps=5000, eval_batch_generator=test_generator, viz_samples=test_samples)
        print('\nTrain loss={}'.format(train_loss))

        test_loss = test(gpt_model, test_generator)
        print('\nTest loss={}'.format(test_loss))

        # scheduler.step()
        print('=' * 80)
    except KeyboardInterrupt:
        print('Training interrupted.')
        break

# TODO сохранять модель после файнтюна для последующего использования.
