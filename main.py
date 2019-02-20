

from lib.util.processing import processing
from lib.model.encoder import SimpleEncoder
from lib.model.decoder import AttentionDecoder

import torch

import random

if __name__ == "__main__":
    

    # 1. Declare the hyperparameter
    device, configure, word_index, index_word, train_loader, test_loader = processing("./configure")

    # print(len(word_index))
    # Declare the encoder model
    model_encoder = SimpleEncoder(configure).to(device)
    model_decoder = AttentionDecoder(configure, device).to(device)


    # Define the optimizer and loss
    criterion = torch.nn.CrossEntropyLoss()
    # encoder optimizer
    optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=configure["lr"])
    optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=configure["lr"])


    # Training
    for epoch in range(configure["epochs"]):
        for idx, item in enumerate(train_loader):

            # transfer to long tensor
            input, target = [i.type(torch.LongTensor).to(device) for i in item]

            if input.size(0) != configure["batch_size"]: continue
            # Encoder   
            encoder_out, encoder_hidden = model_encoder(input)
            
            # Decoder 
            # declare the first input <go>
            decoder_input = torch.tensor([word_index["<go>"]]*configure["batch_size"], 
                                         dtype=torch.long, device=device).view(configure["batch_size"], -1)
            decoder_hidden = encoder_hidden
            z = torch.ones([configure["batch_size"],1,configure["hidden_size"]]).to(device)
            coverage = torch.zeros([configure["batch_size"],configure["max_content"]]).to(device)
            seq_loss = 0
            for i in range(configure["max_output"]):

                decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, input, coverage)

                coverage = coverage

                if random.randint(1, 10) > 5:
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(configure["batch_size"], -1)
                else:
                    decoder_input = target[:,i].view(configure["batch_size"], -1)

                decoder_hidden = decoder_hidden

                step_coverage_loss = torch.sum(torch.min(attn.reshape(-1,1), coverage.reshape(-1,1)), 1) 
                step_coverage_loss = torch.sum(step_coverage_loss)
                # print(coverage)
                # print("---")
                # decoder_output = decoder_output.reshape(configure["batch_size"], -1, 1)
                # print(step_coverage_loss)
                # print((criterion(decoder_output, target[:,i].reshape(configure["batch_size"],-1))))
                # print(-torch.log(decoder_output+target[:,i]))
                seq_loss += (criterion(decoder_output, target[:,i]))

                # print(seq_loss)
        
                seq_loss += step_coverage_loss
      
                # print(decoder_input)
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            seq_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            if (idx) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Coverage Loss: {:4f}' 
                    .format(epoch+1, configure["epochs"], idx, len(train_loader), seq_loss.item(),step_coverage_loss.item()))


        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for idx, item in enumerate(test_loader):
                
                # transfer to long tensor
                input, target = [i.type(torch.LongTensor).to(device) for i in item]
                
                if input.size(0) != configure["batch_size"]: continue
                # Encoder   
                encoder_out, encoder_hidden = model_encoder(input)
                
                # Decoder 
                # declare the first input <go>
                decoder_input = torch.tensor([word_index["<go>"]]*configure["batch_size"], 
                                            dtype=torch.long, device=device).view(configure["batch_size"], -1)
                decoder_hidden = encoder_hidden
                seq_loss = 0
                result = []
                z = torch.ones([configure["batch_size"],1,configure["hidden_size"]]).to(device)
                coverage = torch.zeros([configure["batch_size"],configure["max_content"]]).to(device)
                for i in range(configure["max_output"]):
                    decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, input, coverage)

    
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(configure["batch_size"], -1)


                    decoder_hidden = decoder_hidden

                    total += configure["batch_size"]
                    correct += (torch.max(decoder_output, 1)[1] == target[:,i]).sum().item()
                    # print(torch.max(decoder_output, 1)[1],target[:,i])
                    result.append(index_word[torch.max(decoder_output, 1)[1][1].item()])
                
            with open("test.txt", "a+", encoding="utf-8") as a: a.write("".join(result)+"\n")

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')`


