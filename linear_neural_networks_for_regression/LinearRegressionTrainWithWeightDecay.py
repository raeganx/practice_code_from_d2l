from utils import *
import matplotlib.pyplot as plt


class DataWithHighDimensions(DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)


class LinearRegression_WeightDecay(ModelModule):
    """The linear regression model implemented with high-level APIs.

    Defined in :numref:`sec_linear_concise`"""
    def __init__(self, wd, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)  # specify the output dimension
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        """Defined in :numref:`sec_linear_concise`"""
        return self.net(X)

    def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_concise`"""
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)


def l2_penalty(w):
    return (w ** 2).sum() / 2


model = LinearRegression_WeightDecay(wd=3, lr=0.01) # model instance with weight decay
model.board.yscale='log'

data = DataWithHighDimensions(num_train=20, num_val=100, num_inputs=200, batch_size=5) # designed to be easily overfitted

trainer = Trainer(max_epochs=10)

trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.net.weight.data)))

plt.show()