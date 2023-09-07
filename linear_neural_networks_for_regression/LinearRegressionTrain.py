from utils import *
import matplotlib.pyplot as plt


class SyntheticRegressionData(DataModule):
    """Synthetic data for linear regression.

    Defined in :numref:`sec_synthetic-regression-data`"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, reshape(w, (-1, 1))) + b + noise

    def get_dataloader(self, train):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


class LinearRegression(ModelModule):
    """The linear regression model implemented with high-level APIs.

    Defined in :numref:`sec_linear_concise`"""
    def __init__(self, lr):
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
        """Defined in :numref:`sec_linear_concise`"""
        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        """Defined in :numref:`sec_linear_concise`"""
        return (self.net.weight.data, self.net.bias.data)


model = LinearRegression(lr=0.03)  # model instance without weight decay

data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)  # data instance

trainer = Trainer(max_epochs=3)  # trainer that combines model instance and data instance

trainer.fit(model, data)

estimated_w, estimated_b = model.get_w_b()
print(f'error in estimating w: {data.w - estimated_w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - estimated_b}')

plt.show()
