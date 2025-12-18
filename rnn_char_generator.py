import math
import random
import torch
import torch.nn as nn


def build_dataset(raw_text, device):
    chars = sorted(list(set(raw_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in raw_text], dtype=torch.long, device=device)
    vocab_size = len(stoi)
    return data, stoi, itos, vocab_size


class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=128, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        x = self.emb(x)
        out, h_new = self.rnn(x, h)
        logits = self.head(out)
        return logits, h_new


def make_batch(data, seq_len, batch_size):
    max_start = data.size(0) - seq_len - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in idx])
    return x, y


def sample(model, itos, stoi, device, start_text="hello", length=200, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([[stoi.get(ch, 0) for ch in start_text]], device=device)
        h = None
        generated = list(start_text)
        for _ in range(length):
            logits, h = model(input_ids, h)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            next_ch = itos[next_id.item()]
            generated.append(next_ch)
            input_ids = next_id
    return "".join(generated)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_text = (
        "to be or not to be that is the question whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune or to take arms against a sea of troubles "
        "and by opposing end them"
    )
    data, stoi, itos, vocab_size = build_dataset(raw_text, device)

    seq_len = 64
    batch_size = 32
    lr = 3e-3
    steps = 2000
    print_every = 200

    model = CharRNN(vocab_size, emb_dim=64, hidden_dim=128, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, steps + 1):
        x, y = make_batch(data, seq_len, batch_size)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % print_every == 0:
            print(f"step {step} loss {loss.item():.4f}")

    print("---- sample ----")
    print(sample(model, itos, stoi, device, start_text="to be ", length=200, temperature=0.8))


if __name__ == "__main__":
    main()

