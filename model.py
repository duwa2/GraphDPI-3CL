import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# Define MPNN
class MPNN(MessagePassing):
    def __init__(self, node_feature_dim, edge_feature_dim, message_dim):
        super(MPNN, self).__init__(aggr='add')  # "Add" aggregation.
        # Define the layers here
        self.message_lin = nn.Linear(node_feature_dim + edge_feature_dim, message_dim)
        self.update_lin = nn.Linear(node_feature_dim + message_dim, node_feature_dim)

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Compute the messages
        # x_j refers to the neighboring node features
        tmp = torch.cat([x_j, edge_attr], dim=1)  # Concatenate node features with edge attributes
        return self.message_lin(tmp)

    def update(self, aggr_out, x):
        # Update node features
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.update_lin(tmp)
    
class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0,2,1)
        # conved = [batch size,protein len,hid dim]
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg,dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm,dim=1)
        # norm = [batch size,compound len]
        trg = torch.squeeze(trg,dim=0)
        norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            v = trg[i,]
            v = v * norm[i]
            sum += v
        sum = sum.unsqueeze(dim=0)

        # trg = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label,norm,trg,sum


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input =[num_node, atom_dim]
        # adj = [num_node, num_node]
        support = torch.mm(input, self.weight)
        # support =[num_node,atom_dim]
        output = torch.mm(adj, support)
        # output = [num_node,atom_dim]
        return output

    def forward(self, compound, adj, protein):
        # compound = [atom_num, atom_dim]
        # adj = [atom_num, atom_num]
        # protein = [protein len, 100]
        compound = self.gcn(compound, adj)
        compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1 ,atom_num, atom_dim]

        protein = torch.unsqueeze(protein, dim=0)
        # protein =[ batch size=1,protein len, protein_dim]
        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hid dim]

        out,norm,trg,sum = self.decoder(compound, enc_src)
        # out = [batch size, 2]
        #out = torch.squeeze(out, dim=0)
        return out,norm,trg,sum

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        compound, adj, protein = inputs
        Loss = nn.CrossEntropyLoss()

        if train:
            predicted_interaction,norm = self.forward(compound, adj, protein)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            predicted_interaction,norm,trg,sum = self.forward(compound, adj, protein)
            correct_labels = correct_interaction.to('cpu').data.numpy().item()
            ys = F.softmax(predicted_interaction,1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0,1]
            return predicted_scores,norm,trg,sum

# Define the overall model
class MolecularModel(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, message_dim, lstm_hidden_dim, lstm_layers, fc_output_dim, attention_dim):
        super(MolecularModel, self).__init__()
        self.mpnn = MPNN(node_feature_dim, edge_feature_dim, message_dim)
        self.bilstm = nn.LSTM(input_size=node_feature_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_dim*2, num_heads=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim*2, fc_output_dim)

    def forward(self, data):
        # Apply MPNN
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.mpnn(x, edge_index, edge_attr)
        
        # Apply BiLSTM
        lstm_out, (h_n, c_n) = self.bilstm(x.unsqueeze(0))  # Assuming x is of shape [nodes, features]
        lstm_out = lstm_out.squeeze(0)  # Removing the batch dimension
        
        # Apply self-attention
        attn_output, attn_output_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Apply a fully connected layer
        fc_out = self.fc(attn_output)
        
        return fc_out, attn_output_weights

# Example usage
node_feature_dim = 128
edge_feature_dim = 10
message_dim = 64
lstm_hidden_dim = 256
lstm_layers = 1
fc_output_dim = 1
attention_dim = 256

model = MolecularModel(node_feature_dim, edge_feature_dim, message_dim, lstm_hidden_dim, lstm_layers, fc_output_dim, attention_dim)

