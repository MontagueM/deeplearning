import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# define input size, hidden layer size, output size
D_i, D_k, D_o = 1, 100, 1
# create model with two hidden layers
model = nn.Sequential(
nn.Linear(D_i, D_k),
nn.ReLU(),
nn.Linear(D_k, D_o))
# He initialization of weights
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform(layer_in.weight)
        layer_in.bias.data.fill_(0.0)
model.apply(weights_init)
# choose least squares loss function
criterion = nn.MSELoss()
# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.ASGD(model.parameters(), lr = 0.01)
# object that decreases learning rate by half every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# create 100 random data points and store in data loader class
x = torch.randn(1000, D_i)
# y is a sinusoidal function of x * exponentional of x as an envelope on the sin
y = -2 * x + 5
plt.plot(x.detach().numpy(), y.detach().numpy(), 'o')
plt.show()
data_loader = DataLoader(TensorDataset(x,y), batch_size=10, shuffle=True)
# loop over the dataset 100 times
losses = []
for epoch in range(30):
    epoch_loss = 0.0
    # loop over batches
    for i, data in enumerate(data_loader):
        # retrieve inputs and labels for this batch
        x_batch, y_batch = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        # backward pass
        loss.backward()
        # SGD update
        optimizer.step()
        # update statistics
        epoch_loss += loss.item()
    # print error
    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')
    # tell scheduler to consider updating learning rate
    scheduler.step()
    losses.append(epoch_loss)


# plt.plot(losses)
# plt.show()

# plot predictions
x = torch.randn(100, D_i)
pred = model(x)
plt.plot(x.detach().numpy(), pred.detach().numpy(), 'o')
plt.show()