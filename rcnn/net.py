"""
The RCNN network. Adapted from the network written by 
Prakash Pandey.

https://github.com/prakashpandey9/Text-Classification-Pytorch
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class Net(nn.Module):
    """RCNN network."""
    def __init__(self, batch_size, hidden_size):
        """
        Initialize the network.
        
        Params:
        batch_size: size of the batch 
        hidden_size: size of the hidden_state
        """
        super(Net, self).__init__()
        # Initialize the dimensions of each layer
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = 2 # (deceptive, truthful)
        self.embedding_length = 128 # dimension of the output of the pretrained
                                    # word2vec network
                
        # Initialize the layers in the network
        self.lstm = nn.LSTM(input_size=self.embedding_length, hidden_size=self.hidden_size, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embedding_length, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        # Initialize the max pool function
        self.pool = F.max_pool1d
        


    def forward(self, seq_tensor, batch_size=None):
        """
        Processor the input seq_tensor of shape (batch_size, num_sequences, embedding_length).
        
        Params:
        seq_tensor: the input tensor
        batch_size: default = None. Used only for 
                    prediction on a single sentence (batch_size = 1)
        """
        seq_tensor = seq_tensor.permute(1, 0, 2)

        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
        
        
        # Concatenate the embedding of the sequence
        # with its left and right contextual embedding.
        out_lstm, (final_hidden_state, final_cell_state) = self.lstm(seq_tensor, (h_0, c_0))
        final_encoding = torch.cat((out_lstm, seq_tensor), 2).permute(1, 0, 2)
        
        # Map the long concatednated encoding vector
        # back to the size of the hidden_size
        fc_out =  self.fc(final_encoding) # fc_out.size() = (batch_size, num_sequences, hidden_size)
 
        fc_out = fc_out.permute(0, 2, 1) 
        
        # Use a max pooling layer across all sequences of texts.
        conv_out = self.pool(fc_out, fc_out.size()[2]) # conv_out.size() = (batch_size, hidden_size, 1)
        
        
        conv_out = conv_out.squeeze(2)
        # Map to the output layer
        logits = self.out(conv_out)
        return logits