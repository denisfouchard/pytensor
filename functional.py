from numbers import Number
import math
from tensor import Tensor

_sum = sum


def exp(x: Tensor) -> Tensor:
    return x.apply_element(lambda z: math.exp(z))


def log(x: Tensor) -> Tensor:
    return x.apply_element(math.log)


def softmax(x: Tensor) -> Tensor:
    assert x.dim == 1 or x.dim == 2
    if x.dim == 1:
        x_ = exp(x)
        return x_ / x_.sum()
    if x.dim == 2:
        data = [softmax(Tensor(y)).data for y in x.data]
        return Tensor(data=data)


if __name__ == "__main__":
    k = Tensor([[1, 10, 19], [1, 10, 3]])
    # print(k.apply_element(lambda x: x**2))
    # print(exp(k))
    print(softmax(k))
