

import torch 

class SimpleEncoder(torch.nn.Module):

    def __init__(self, configure):
        super(SimpleEncoder, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=configure["num_words"], 
                                            embedding_dim=configure["embedding_dim"])

        self.gru = torch.nn.LSTM(input_size=configure["embedding_dim"],
                                hidden_size=configure["hidden_size"],
                                num_layers=configure["num_layers"],
                                bidirectional=False,
                                batch_first=True)

        self.fc = torch.nn.Linear(configure["hidden_size"], configure["num_words"])



    def forward(self, input):

        # Embedding
        embedding = self.embedding(input)

        # Call the GRU
        out, hidden = self.gru(embedding)

        return out, hidden
