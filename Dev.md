## Installing dependencies

To setup the environment, you need to be in specific Python versions.
At least you won't pip install successfully in Python 3.11 .
And README says it works with Python 3.8.x to 3.10.x, so let's use Python 3.9 .

```bash
pyenv install 3.9
pyenv virtualenv 3.9 neu39
pyenv activate neu39

pip install -r requirements.txt
```

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
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```


### Chapter 2

```python
import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
```
