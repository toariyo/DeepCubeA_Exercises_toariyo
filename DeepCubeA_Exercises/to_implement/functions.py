
import torch
from torch import nn, optim
import numpy as np
from typing import List
from numpy import testing
from torch import nn, tensor
import numpy as np
from torch._C import device
from environments.environment_abstract import Environment, State
import torch

class nnet_model(nn.Module):
    def __init__(self):
        super(nnet_model, self).__init__()
        self.linLayer1 = torch.nn.Linear(81, 39)
        self.reluLayer1 = torch.nn.ReLU()
        self.linLayer2 = torch.nn.Linear(39,30)
        self.reluLayer2 = torch.nn.ReLU()
        self.linLayer3 = torch.nn.Linear(30, 1)
        self.reluLayer3 = torch.nn.ReLU()
    def forward(self, x):
        x = self.linLayer1(x)
        x = self.reluLayer1(x)
        x = self.linLayer2(x)
        x = self.reluLayer2(x)
        x = self.linLayer3(x)
        x = self.reluLayer3(x)
        return x

def get_nnet_model() -> nn.Module:
    """ Get the neural network model
    @return: neural network model
    """
    return nnet_model()
pass


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int, device: torch.device, lr=0.01, lr_d=1):
    input_tensor = tensor(states_nnet)  # Create tensors for inputs and outputs
    target = tensor(outputs)

    criterion = nn.MSELoss()  # Define criterion function

    batchsize = 100

    input_tensor = torch.from_numpy(states_nnet).float()  # Tensors converted to float value
    target = torch.from_numpy(outputs).float()

    optimizer = torch.optim.SGD(nnet.parameters(), lr=0.01) # Define optimizer function

    data = torch.utils.data.TensorDataset(input_tensor, target)
    # Create dataLoader
    data_loader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=True, num_workers=4)


    # training loop
    for epoch, data in enumerate(data_loader, 0):
        inputs, outputs = data # Place inputs and outputs within data variable
        nnetOutput = nnet(inputs) # nnetoutputs is initialized based on inputs
        optimizer.zero_grad()
        loss = criterion(nnetOutput, outputs) # Loss function is utilized
        loss.backward()
        optimizer.step()
    pass

pass

def value_iteration(nnet, device, env: Environment, states: List[State]) -> List[float]:
    env = nnet

    qtable = np.random.rand()

    epochs = 100
    epsilon = 0.08
    done = False
    for i in range(epochs):
        steps = 0
        while not done:
            print('epoch #', i + 1, epochs)
            steps += 1
            states = tensor(states).float()
            if np.random.uniform() < epsilon:
                action = env.randomAction()

            else:
                action = qtable[states].index(max(qtable[states]))

            next_state, reward, done = env.step(action)

            qtable[states][action] = reward + 0.1 * max(qtable[next_state])

            state = next_state

            epsilon -= 0.1 * epsilon

            print("\nDone in", steps, "steps".format(steps))

    pass