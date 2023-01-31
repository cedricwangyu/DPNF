import torch
from torch import nn

torch.set_default_tensor_type(torch.DoubleTensor)


def train(model, x, optimizer, i, args, loglist, device=None):
    model.train()
    if device is not None: x = x.to(device)
    loss = - model.log_prob(x).mean(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Iteration: {}\tLoss: {}'.format(i, loss.item()), end='\r')
    if i % args.log_interval == 0:
        loglist.append(loss.item())
        print('Iteration: {}\tLoss: {}'.format(i, loss.item()))


def train_noisy(model, x, optimizer, i, args, loglist, device=None):
    model.train()
    if device is not None: x = x.to(device)
    grads = [0 for _ in model.parameters()]
    log_probs = - model.log_prob(x)
    n = log_probs.size(0)
    for j, lp in enumerate(log_probs):
        optimizer.zero_grad()
        if j < n - 1:
            lp.backward(retain_graph=True)
        else:
            lp.backward()
        if args.noisy == 'True':
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=exp.C, norm_type=2)
        for jj, p in enumerate(model.parameters()):
            grads[jj] += p.grad.clone()

    optimizer.zero_grad()
    if args.noisy == 'True':
        for j, p in enumerate(model.parameters()):
            p.grad = (grads[j] + torch.randn(p.grad.size(), device=device) * args.C * args.sigma) / n
    else:
        for j, p in enumerate(model.parameters()):
            p.grad = grads[j] / n

    optimizer.step()
    loss = log_probs.mean()
    print('Iteration: {}\tLoss: {}'.format(i, loss.item()), end='\r')
    if i % args.log_interval == 0:
        loglist.append(loss.item())
        print('Iteration: {}\tLoss: {}'.format(i, loss.item()))
