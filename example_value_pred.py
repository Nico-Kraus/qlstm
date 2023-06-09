import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lstm_tagger import LSTMTagger

from matplotlib import pyplot as plt

data = [1, 2, 3, 1, 2, 3, 1, 2, 3]
training_data = []
lookback = 3
for i in range(len(data) - lookback):
    training_data.append([data[i : i + lookback], data[i + 1 : i + lookback + 1]])

if __name__ == "__main__":
    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    parser = argparse.ArgumentParser("QLSTM Example")
    parser.add_argument("-E", "--embedding_dim", default=8, type=int)
    parser.add_argument("-H", "--hidden_dim", default=6, type=int)
    parser.add_argument("-Q", "--n_qubits", default=2, type=int)
    parser.add_argument("-e", "--n_epochs", default=500, type=int)
    parser.add_argument("-B", "--backend", default="default.qubit")
    args = parser.parse_args()

    print(f"Embedding dim:    {args.embedding_dim}")
    print(f"LSTM output size: {args.hidden_dim}")
    print(f"Number of qubits: {args.n_qubits}")
    print(f"Training epochs:  {args.n_epochs}")

    model = LSTMTagger(
        args.embedding_dim,
        args.hidden_dim,
        input_size=max(data) + 1,
        tagset_size=1,
        n_qubits=args.n_qubits,
        backend=args.backend,
    )

    loss_function = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    history = {"loss": [], "acc": []}
    for epoch in range(args.n_epochs):
        losses = []
        preds = []
        targets = []
        for x_val, y_val in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors
            input = torch.tensor(x_val, dtype=torch.long)
            labels = torch.tensor(y_val, dtype=torch.long)

            # Step 3. Run our forward pass.
            tag_scores = model(input)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores[-1], labels[-1])
            loss.backward()
            optimizer.step()
            losses.append(float(loss))

            probs = torch.tensor(list(tag_scores))
            preds.append(probs)
            targets.append(labels)

        avg_loss = np.mean(losses)
        history["loss"].append(avg_loss)

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        corrects = torch.tensor(
            [np.isclose(preds[i], targets[i], atol=0.1) for i in range(len(targets))]
        )
        accuracy = corrects.sum().float() / float(targets.size(0))
        history["acc"].append(accuracy)

        print(
            f"Epoch {epoch+1} / {args.n_epochs}: Loss = {avg_loss:.3f} Acc = {accuracy:.2f}"
        )

    # See what the scores are after training
    with torch.no_grad():
        for x_val, y_val in training_data:

            tag_scores = model(torch.tensor(x_val, dtype=torch.long))

            pred = tag_scores.flatten().tolist()

            print(f"Input:  {x_val}")
            print(f"Output:    {y_val}")
            print(f"Predicted: {pred}")

    lstm_choice = "classical" if args.n_qubits == 0 else "quantum"

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(history["loss"], label=f"{lstm_choice} LSTM Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(history["acc"], label=f"{lstm_choice} LSTM Accuracy", color="tab:red")

    plt.title("Part-of-Speech Tagger Training")
    plt.ylim(0.0, 1.5)
    plt.legend(loc="upper right")

    plt.savefig(f"training_{lstm_choice}.png")
    plt.show()
