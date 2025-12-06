class Tensor:
    # When initalised...
    def __init__(self,data):
        self.data = data
        self.grad = 0

    # Return on multiplication...
    def __mul__(self, other):
        return Tensor(self.data * other)
    
    # Right-hand-side multiplication
    def __rmul__(self, other):
        return (self.data * other)
    
    # Left-hand and right-hand subtraction
    def __sub__(self, other):
        return (self.data - other)

    def __rsub__(self, other):
        return (other - self.data)
    
  # Left-hand and right-hand power
    def __pow__(self, other):
        return (self.data ** other)
    def __rpow__(self, other):
        return (other ** self.data )