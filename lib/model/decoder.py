

import torch 
from .learning_attention import Attention 

class SimpleDecoder(torch.nn.Module):

    def __init__(self, configure):
        super(SimpleDecoder, self).__init__()

        # Declare the hyperparameter
        self.configure = configure

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=configure["num_words"], 
                                            embedding_dim=configure["embedding_dim"])

        self.gru = torch.nn.GRU(input_size=configure["embedding_dim"],
                                hidden_size=configure["hidden_size"],
                                num_layers=configure["num_layers"],
                                bidirectional=False,
                                batch_first=True)

        self.fc = torch.nn.Linear(configure["hidden_size"], configure["num_words"])



    def forward(self, input, hidden):

        # Embedding
        embedding = self.embedding(input)

        # Call the GRU
        out, hidden = self.gru(embedding, hidden)

        out = self.fc(out.view(out.size(0),-1))

        return out, hidden



class AttentionDecoder(torch.nn.Module):
    
    def __init__(self, configure, device):
        super(AttentionDecoder, self).__init__()

        # Declare the hyperparameter
        self.configure = configure
        self.device = device
        self.configure = configure

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=configure["num_words"], 
                                            embedding_dim=configure["embedding_dim"])

        self.gru = torch.nn.LSTM(input_size=configure["embedding_dim"]+configure["hidden_size"],
                                hidden_size=configure["hidden_size"],
                                num_layers=configure["num_layers"],
                                bidirectional=False,
                                batch_first=True)

        self.att = Attention(configure["hidden_size"])

        self.fc = torch.nn.Linear(configure["hidden_size"], configure["num_words"])

        self.p = torch.nn.Linear(configure["batch_size"]*2+configure["embedding_dim"], 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, hidden, encoder_output, z, content, coverage):

        # Embedding
        embedding = self.embedding(input)
        # print(embedding.squeeze().size())

        combine = torch.cat([embedding,z],2)
        # print(combine.squeeze().size())
        # Call the GRU
        out, hidden = self.gru(combine, hidden)

        # call the attention
        output, attn, coverage = self.att(output=out, context=encoder_output, coverage=coverage)
        

        index = content
        attn = attn.view(attn.size(0),-1)
        attn_value = torch.zeros([attn.size(0), self.configure["num_words"]]).to(self.device)
        attn_value = attn_value.scatter_(1, index, attn)

        out = self.fc(output.view(output.size(0),-1))

        p = self.sigmoid(self.p(torch.cat([embedding.squeeze(), combine.squeeze()], 1)))
        # print(p)
        out = (1-p)*out + p*attn_value
        # print(attn_value.size(), output.size())

        return out, hidden, output, attn, coverage
