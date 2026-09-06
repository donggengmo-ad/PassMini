from scripts.tokenizer import CharTokenizer
from scripts import data
from scripts.models import AutoregressiveGRU
import torch
import math
def test_gru():
    word_list = ["hello", "world", "this", "is", "a", "test"]
    token = CharTokenizer.from_text(word_list)
    dataset = data.PasswordDataset(tokenizer=token, passwords=word_list)
    num_layers = 2
    hidden_size = 128
    model = AutoregressiveGRU(token,
                             num_layers=num_layers,
                             hidden_size=hidden_size)
    # +1 for input BOS and target EOS
    sequence_length = max(len(token.encode(word)) for word in word_list) + 1
    batch_size = 4
    batch_list = [dataset[i] for i in range(batch_size)]
    assert model.gru.batch_first is True
    input_batch, target_batch = data.collate_batch(batch_list, pad_id=token.pad_id)
    logits, hidden = model(input_batch, return_hidden=True)
    assert logits.shape == (
        batch_size,
        sequence_length,
        token.vocab_size,
    )
    assert hidden.shape == (
        num_layers,
        batch_size,
        hidden_size,
    )
    assert logits.dtype == torch.float
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=token.pad_id,
    )
    loss = loss_fn(logits.reshape(-1, token.vocab_size), target_batch.reshape(-1))
    loss.backward()
    assert torch.all(torch.isfinite(logits))
    assert torch.isfinite(loss)

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.all(torch.isfinite(parameter.grad)), name

    pad_grad = model.embedding.weight.grad[token.pad_id]
    assert torch.all(pad_grad == 0)

    non_pad_grad = model.embedding.weight.grad[token.bos_id]
    assert torch.any(non_pad_grad != 0)

def test_gru_can_overfit_one_sequence():
    torch.manual_seed(42)
    word_list = ["hello"]

    token = CharTokenizer.from_text(word_list)
    dataset = data.PasswordDataset(tokenizer=token, passwords=word_list)
    batch_size = 5
    batch_list = [dataset[0] for _ in range(batch_size)]
    input_batch, target_batch = data.collate_batch(batch_list, pad_id=token.pad_id)

    model = AutoregressiveGRU(token,
                             num_layers=1,
                             embedding_dim=16,
                             hidden_size=32)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=token.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    epochs = 200
    init_final_loss = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(input_batch)
        loss = criterion(logits.reshape(-1, token.vocab_size), target_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            l = loss.item()
            assert math.isfinite(l)
            assert l >= 0
            if epoch == 0:
                init_final_loss.append(l)

    model.eval()
    with torch.no_grad():
        loss = criterion(model(input_batch).reshape(-1, token.vocab_size), target_batch.reshape(-1))
        init_final_loss.append(loss.item())
        assert init_final_loss[-1] < 0.1
        assert init_final_loss[-1] < init_final_loss[0]

        inputs, targets = dataset[0]
        logits = model(inputs.unsqueeze(0))
        predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)
        mask = targets != token.pad_id
        assert torch.equal(predicted_ids[mask], targets[mask])
