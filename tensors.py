class Tensor:
    def __init__(self,data):
        self.data = data
        self.grad = 0