from backgrad import Value
import torch

x = Value(3)
y = Value(5)
z = Value(2)
n = Value(7)

w = x / (2*(y**3+1)**2 - 4*n**2) + z.relu()
w.backprop()

print(f"Mine: x: '{x}', y: '{y}', n: '{n}', z: '{z}', w: {w.val}")

xt = torch.Tensor([3.0]).double()
yt = torch.Tensor([5.0]).double()
zt = torch.Tensor([2.0]).double()
nt = torch.Tensor([7.0]).double()
xt.requires_grad = True
yt.requires_grad = True
zt.requires_grad = True
nt.requires_grad = True
wt = xt / (2*(yt**3+1)**2 - 4*nt**2) + zt.relu()
wt.backward()
print(f"Torch: xt: '{xt.grad[0]}', yt: '{yt.grad[0]}', nt: '{nt.grad[0]}', zt: '{zt.grad[0]}', wt: {wt.data[0]}")