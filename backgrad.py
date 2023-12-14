from enum import Enum


class Value:
    def __init__(self, val, children=[]):
        self.val = val
        self.grad = 1
        self.children = children

    def __repr__(self):
        return f"Value(val={self.val}, grad={self.grad})"

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return MultiplyValue(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return SubtractValue(self, other)

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return DivideValue(self, other)

    def __floordiv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return DivideValue(self, other)

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return AddValue(self, other)

    def __pow__(self, power, modulo=None):
        return PowerValue(self, power)

    def relu(self):
        return ReluValue(self)

    def calc_grad(self):
        pass

    def backprop(self):
        for child in self.children:
            if not child.children:
                continue
            child.calc_grad()
            child.backprop()


class ReluValue(Value):
    def __init__(self, a):
        relu_val = a.val if a.val > 0 else 0
        super().__init__(relu_val, [a])
        self.a = a

    def calc_grad(self):
        # ReLU grad is 0 if <= 0, 1 > 0
        print(f"ReLU {self.a.val}")
        if self.a.val > 0:
            self.a.grad = 1
        else:
            self.a.grad = 0


class MultiplyValue(Value):
    def __init__(self, a, b):
        super().__init__(a.val * b.val, [a, b])
        self.a = a
        self.b = b

    def calc_grad(self):
        # product rule
        self.a.grad = self.grad * self.b.val
        self.b.grad = self.grad * self.a.val


class DivideValue(Value):
    def __init__(self, a, b):
        super().__init__(a.val / b.val, [a, b])
        self.a = a
        self.b = b

    def calc_grad(self):
        # quotient rule
        self.a.grad = self.grad / self.b.val
        self.b.grad = -self.grad*self.a.val / self.b.val**2


class AddValue(Value):
    def __init__(self, a, b):
        super().__init__(a.val + b.val, [a, b])
        self.a = a
        self.b = b

    def calc_grad(self):
        self.a.grad = self.grad
        self.b.grad = self.grad


class SubtractValue(Value):
    def __init__(self, a, b):
        super().__init__(a.val - b.val, [a, b])
        self.a = a
        self.b = b

    def calc_grad(self):
        self.a.grad = self.grad
        self.b.grad = self.grad


class PowerValue(Value):
    def __init__(self, a, power):
        super().__init__(a.val ** power, [a])
        self.a = a
        self.power = power

    def calc_grad(self):
        self.a.grad = self.grad * (self.power * self.a.val ** (self.power-1))
