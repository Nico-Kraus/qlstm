import torch.nn as nn
import torch.nn.functional as F
from qlstm_pennylane import QLSTM


class LSTMTagger(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        input_size,
        tagset_size,
        n_qubits=0,
        backend="default.qubit",
    ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if n_qubits > 0:
            print(f"Tagger will use Quantum LSTM running on backend {backend}")
            self.lstm = QLSTM(
                embedding_dim, hidden_dim, n_qubits=n_qubits, backend=backend
            )
        else:
            print("Tagger will use Classical LSTM")
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input):
        embeds = self.embeddings(input)
        lstm_out, _ = self.lstm(embeds.view(len(input), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(input), -1))
        tag_scores = F.log_softmax(tag_logits, dim=1)
        return tag_scores
