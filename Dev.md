## Installing dependencies

To setup the environment, you need to be in specific Python versions.
At least you won't pip install successfully in Python 3.11 .
And README says it works with Python 3.8.x to 3.10.x, so let's use Python 3.9 .

It's recommended to use uv
```
uv sync
```

Otherwise, you can setup environment manually

```bash
pyenv install 3.9
pyenv virtualenv 3.9 neu39
pyenv activate neu39

pip install -r requirements.txt
```

## Cuda

The code in this project is not written to use GPU. So it is not actually using GPU. So Cuda is not required.

But if you do want to install Cuda for torch, install Cuda 12.4. Not 12.6, not 12.8. (I can't get torch 2.6.0 to work with 12.6)

12.4 is well supported by uv.


## Running

```bash
cd src
```

### Chapter 1

```python
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
num_epochs = 30
learning_rate = 3.0
net.SGD(training_data, num_epochs, 10, learning_rate, test_data=test_data)
```


### Chapter 2

```python
import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
num_epochs = 30
learning_rate = 0.5
net.SGD(training_data, num_epochs, 10, learning_rate, evaluation_data=test_data, monitor_evaluation_accuracy=True)
```
