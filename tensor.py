from typing import Iterable, Tuple, List, Callable, Optional
from numbers import Number
from copy import deepcopy

_sum = sum


def _one_dim_dot(first: Iterable[Number], second: Iterable[Number]):
    assert len(first) == len(second)
    return sum(x * y for x, y in zip(first, second))


class Tensor:
    data: Iterable | Number
    size: List[int]
    dim: int

    def __init__(self, data: Iterable | Number):
        self.data = data
        if isinstance(data, Number):
            self.size = []
            self.dim = 0
            return
        _size = [len(self.data)]
        _data = data
        while isinstance(_data[0], Iterable):
            _l = len(_data[0])
            for x in _data:
                if len(x) != _l:
                    raise IndexError(
                        f"Expected all items to be of size {_l}, but got {len(x)}."
                    )
            _size.append(_l)
            _data = _data[0]

        self.size = _size
        self.dim = len(_size)

    # Data Structure Operations
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: int | Tuple[int]) -> "Tensor":
        if isinstance(key, int):
            return Tensor(self.data[key])
        if len(key) > len(self.size):
            raise IndexError()
        _data = self.data
        for i, k in enumerate(key):
            if i == len(key) - 1:
                return Tensor(_data[k])
            else:
                _data = _data[k]

        return _data

    def __setitem__(self, key: int | Tuple[int], element: Number):
        assert self.dim > 0
        if isinstance(key, int):
            assert self.dim == 1
            self.data[key] = element
            return
        assert len(key) == self.dim
        _data = self.data
        for i, k in enumerate(key):
            if i == len(key) - 1:
                _data[k] = element
                return
            else:
                _data = _data[k]
        return

    def __repr__(self):
        return f"Tensor({self.data}, size={self.size})"

    def copy(self):
        return Tensor(data=deepcopy(self.data))

    # Numerical Operations
    def __eq__(self, element: "Tensor"):
        return self.data == element.data

    def __sub__(self, element: "Tensor") -> "Tensor":
        assert element.size == self.size
        if self.dim == 0:
            return Tensor(self.data - element.data)
        data = [(Tensor(x) - Tensor(y)).data for x, y in zip(self.data, element.data)]
        return Tensor(data)

    def __add__(self, element: "Tensor") -> "Tensor":
        assert element.size == self.size
        if self.dim == 0:
            return Tensor(self.data + element.data)
        data = [(Tensor(x) + Tensor(y)).data for x, y in zip(self.data, element.data)]
        return Tensor(data)

    def __mul__(self, element: "Tensor"):
        if element.dim == 0:
            return element.data * self
        if self.dim == 0:
            return self.data * element
        if self.dim == element.dim - 1:
            assert self.dim == element.dim[-1:]
            data = [(self * Tensor(y)).data for y in element.data]
        assert self.size == element.size
        data = [(Tensor(x) * Tensor(y)).data for x, y in zip(self.data, element.data)]
        return Tensor(data)

    def __rmul__(self, scalar: Number) -> "Tensor":
        if self.dim == 0:
            return Tensor(scalar * self.data)
        data = [(scalar * Tensor(x)).data for x in self.data]
        return Tensor(data)

    def __truediv__(self, scalar: Number) -> "Tensor":
        if self.dim == 0:
            return Tensor(self.data / scalar)
        data = [(Tensor(x) / scalar).data for x in self.data]
        return Tensor(data)

    def __neg__(self):
        if self.dim == 0:
            return Tensor(-self.data)
        data = [(-Tensor(y)).data for y in self.data]
        return Tensor(data)

    def transpose(self) -> "Tensor":
        assert self.dim == 2
        n, m = self.size
        transposed = Tensor.zeros(size=(m, n)).data
        for i in range(n):
            for j in range(m):
                transposed[j][i] = self.data[i][j]

        return Tensor(data=transposed)

    @property
    def T(self):
        return self.transpose()

    def dot(self, element: "Tensor"):
        if self.dim == 1 and element.dim == 1:
            return Tensor(_one_dim_dot(self.data, element.data))

        if self.dim == 1 and element.dim == 2:
            _data = [self.dot(Tensor(x)).data for x in element.T.data]
            return Tensor(data=_data)

        if self.dim == 2 and element.dim == 1:
            _data = [element.dot(Tensor(x)).data for x in self.T.data]
            return Tensor(_data)

        if self.dim == 2 and element.dim == 2:
            _data = [self.dot(Tensor(x)).data for x in self.T.data]
            return Tensor(data=_data)

    @staticmethod
    def zeros(size: int | Tuple[int]) -> "Tensor":
        if isinstance(size, int):
            return Tensor(data=[0 for _ in range(size)])
        else:
            _size = list(size)
            _data = list(0 for _ in range(_size[-1]))
            for dim_i in list(reversed(_size))[1:]:
                _data = list(deepcopy(_data) for _ in range(dim_i))
            return Tensor(data=_data)

    @staticmethod
    def zeros_like(x: "Tensor") -> "Tensor":
        return Tensor.zeros(size=x.size)

    def apply_element(self, func: Callable[Number, Number]):
        if self.dim == 0:
            return Tensor(data=func(self.data))
        data = [Tensor(x).apply_element(func=func).data for x in self.data]
        return Tensor(data)

    def sum(self) -> Number:
        if self.dim == 0:
            return self.data
        if self.dim == 1:
            return _sum(self.data)

        return _sum(Tensor(y).sum() for y in self.data)

    def mean(self, dim: Optional[int]) -> Number:
        if dim.is_integer():
            assert dim < self.dim
            raise NotImplementedError()

        N = 1
        for d in self.size:
            N *= d
        return self.sum() / N


class GradTensor(Tensor):
    grad: "Tensor"
    requires_grad: bool
    is_leaf: bool

    def __init__(self, data: Iterable | Number, requires_grad: bool = True):
        super(GradTensor, self).__init__(data=data)
        self.requires_grad = requires_grad

    def __add__(self, element):
        return Tensor.__add__(self=self, element=element)


if __name__ == "__main__":
    k = Tensor([[1, 2, 1999], [1, 3, 3]])
    y = Tensor([[1, 2, 3], [1, 3, 3]])
    print(y.mean())
    # print(k.apply_element(lambda x: 2 * x))
    # print(k / 2)
    # print(k + y)
    # print(k - y)
    # print(-y)
    # print(3 * y)
    # k[1, 1] = 199
    # print(k)
    # print(Tensor.zeros((2, 3)))
    # print(k.transpose())
    # print(k.T)
    # print(k.dot(k.T))
    # k = Tensor(1)
    # y = Tensor(1)
    # print(k + y)
    # print(k - y)
    # print(-y)
    # print(3 * y)
