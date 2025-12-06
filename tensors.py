class Tensor:
    # When initalised...
    def __init__(self,data, requires_grad=False):
        self.data = data
        self.grad = 0
        self.require_grad = requires_grad

    # Return on multiplication...
    def __mul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data * other_data, requires_grad=self.require_grad)
    
    # Right-hand-side multiplication
    def __rmul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data * other_data, requires_grad=self.require_grad)
    
    # Left-hand and right-hand subtraction
    def __sub__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data - other_data, requires_grad=self.require_grad)
    def __rsub__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(other_data - self.data, requires_grad=self.require_grad)
    
  # Left-hand and right-hand power
    def __pow__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data ** other_data, requires_grad=self.require_grad)
    def __rpow__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(other_data ** self.data, requires_grad=self.require_grad)