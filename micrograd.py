
"""From Andrej Kaparthy's"""

class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += result.grad 
            other.grad += result.grad 
        
        result._backward = _backward
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        
        result._backward = _backward
        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        result = Value(self.data**other, (self,))

        def _backward():
            self.grad += (other * self.data**(other-1)) * result.grad

        result._backward = _backward

        return result

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        result = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * result.grad
        
        result._backward = _backward

        return result

    def exp(self):
        x = self.data
        result = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += result.data * result.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        
        result._backward = _backward
        return result
    
    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        
        self.grad = 1.0

        print(topo)

        for node in reversed(topo):
            node._backward()


m = Value(3)
x = Value(2)
b = Value(214124)

y = m*x + b

y.backward() # Do backward propogation to calculate gradients
print(x.grad) # NOTE: this will be equivilant to m


