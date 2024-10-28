import pandas as pd
import numpy as np
import nltk
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# load in word tokenized data
with open("list_of_lists.pkl", "rb") as file:
    loaded_list_of_lists = pickle.load(file)
    
# load in labels
    
# encode y labels
labelencoder = LabelEncoder()
y = list(text_df['emotions'])
y = labelencoder.fit_transform(y)

# get embedding look up table for embedding layer
embed_lookup = KeyedVectors.load_word2vec_format('wword2vec.model', binary = False)

# get embedding idx for each word in each text
def get_embed_idx(embed_lookup, data_by_words):

    embed_idx = []
    for word_doc in data_by_words:
        ints = []
        for word in word_doc:
            try:
                idx = embed_lookup.vocab[word].index
            except: 
                idx = 0
            ints.append(idx)
        embed_idx.append(ints)
    
    return embed_idx

embed_indexed_texts = get_embed_idx(embed_lookup, data_by_words)
vocab_size = len(embed_lookup.vocab)

# to make sure inputs are consistent lengtgs
def pad_features(embed_indexed_texts, seq_length):
    
    # getting the correct rows x cols shape
    features = np.zeros((len(embed_indexed_texts), seq_length), dtype=int)

    for i, row in enumerate(embed_indexed_texts):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

seq_length = 200
X_features = pad_features(embed_indexed_texts, seq_length)

# split into train, val, test
X_train, X_val_test, y_train, y_val_test = train_test_split(X_features, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.50)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

# set batch size
batch_size = 50

# shuffling and batching data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# Hyperparameters
num_classes = 6
num_filters = 100
dropout = 0.5
lr = 0.0001

# Text CNN Sentiment Classifier
class TextCNN(nn.Module):
    def __init__(self, embed_model, embedding_dim, vocab_size,
                 num_filters, num_classes, dropout, kernel_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))
        self.convs = nn.ModuleList(
            nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding = (k - 2,0)) 
            for k in kernel_sizes])
        )
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
    
    def conv_and_pool(self, x, conv):
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        x = F.relu(conv(x)).squeeze(3)
        
        # 1D pool over conv_seq_length
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max
    
    def forward(self, x):
        # embedded vectors
        embeds = self.embedding(x) 
        embeds = embeds.unsqueeze(1)
        
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        
        # final logit
        logit = self.fc(x) 
        
        # sigmoid-activated --> a class score
        return self.sig(logit)

# Instantiate the model
net = TextCNN(
    embed_model=embed_lookup,
    embedding_dim=100,
    vocab_size=vocab_size,
    num_filters=num_filters,
    num_classes=num_classes,
    dropout=dropout,
    kernel_sizes=[3,4,5]
)

# Loss and Optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training loop
def train(net, train_loader, epochs, print_every=100):

    counter = 0 
    
    # train for some number of epochs
    net.train()
    for e in range(epochs):

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            inputs, labels = inputs.to(device), labels.to(device)

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output = net(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    output = net(inputs)
                    val_loss = criterion(output, labels)

                    val_losses.append(val_loss.item())

                net.train()
                print(f"Epoch: {e+1}/{epochs}...",
                      f"Step: {counter}...",
                      f"Loss: {loss.item()}...",
                      f"Val Loss: {np.mean(val_losses)}")

# training params
epochs = 50
print_every = 100

train(net, train_loader, epochs, print_every=print_every)

# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    inputs, labels = inputs.to(device), labels.to(device)
    
    # get predicted outputs
    output = net(inputs)
    
    # calculate loss
    test_loss = criterion(output, labels)
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class, get max prob class
    pred = torch.argmax(output, dim=1) 
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels)
    correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# avg test loss
print(f"Test loss: {np.mean(test_losses)}")

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print(f"Test accuracy: {test_acc}")