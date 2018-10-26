import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def one_hot(element_idx):
    x = torch.zeros(n_elements).float()
    x[element_idx] = 1.0
    return x


def generate_embedding(
        pair_data,\
        n_elements,\
        embedding_dims = 5, \
        learning_rate = 0.001, n_epoch=100):
    """
    """

    W1 = Variable(torch.randn(n_elements, embedding_dims).float(), requires_grad=True)
    W2 = Variable(torch.randn(embedding_dims, n_elements).float(), requires_grad=True)

    #this is stochastic gradient descent
    for data,target in pair_data:
        x = Variable(one_hot(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
    
    loss.backward()
        W1.data -= learning_rate*W1.grad.data 
        W2.data -= learning_rate*W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    return W1, W2


