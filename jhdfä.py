import torch

actions = torch.Tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

actions = actions.split((2, 3), dim=1)
print(actions)
actions = [a.argmax(dim=1) for a in actions]
print(actions)
