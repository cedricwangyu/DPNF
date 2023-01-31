import torch
import numpy as np
from torch import nn

torch.set_default_tensor_type(torch.DoubleTensor)


def train(nf, model, optimizer, iteration, args, log, ind=None, sampling=True, surrogate=False):
    nf.train()
    x0 = nf.base_dist.sample([args.batch_size])
    xk, sum_log_abs_det_jacobians = nf(x0)
    if sampling and iteration % 200 == 0:
        x00 = nf.base_dist.sample([args.n_sample])
        xkk, _ = nf(x00)
        np.savetxt(args.output_dir + '/samples' + str(iteration), xkk.cpu().data.numpy(), newline="\n")

    if torch.any(torch.isnan(xk)):
        print("Error Iteration " + str(iteration))
        print(xk)
        np.savetxt(args.output_dir + '/' + args.log_file, np.array(log), newline="\n")
        return False
    optimizer.zero_grad()

    if ind is None:
        loss = (- torch.sum(sum_log_abs_det_jacobians, dim=1, keepdim=True) - model.den_t(xk,
                                                                                          surrogate=surrogate)).mean()
    else:
        loss = (- torch.sum(sum_log_abs_det_jacobians, dim=1, keepdim=True) * args.poisson_ratio - torch.sum(
            model.den_t(xk, ind=ind, surrogate=surrogate), dim=1, keepdim=True)).mean()
    loss.backward()
    print("Iteration: {}\tLoss: {}".format(iteration, loss.item()), end='\r')
    if iteration % args.log_interval == 0:
        print("Iteration: {}\tLoss: {}".format(iteration, loss.item()))
        log.append([iteration, loss.item()])

    optimizer.step()


def train_noisy(nf, model, optimizer, device, iteration, args, log, ind=None, sampling=True, surrogate=False):
    grads = [0 for _ in nf.parameters()]
    nf.train()
    x0 = nf.base_dist.sample([args.batch_size])
    xk, sum_log_abs_det_jacobians = nf(x0)
    if sampling and iteration % 200 == 0:
        x00 = nf.base_dist.sample([args.n_sample])
        xkk, _ = nf(x00)
        np.savetxt(args.output_dir + '/samples' + str(iteration), xkk.cpu().data.numpy(), newline="\n")

    if torch.any(torch.isnan(xk)):
        print("Error Iteration " + str(iteration))
        print(xk)
        np.savetxt(args.output_dir + '/' + args.log_file, np.array(log), newline="\n")
        return False

    LogJoint = (- model.den_t(xk, ind=ind, surrogate=surrogate)).mean(0) - torch.sum(sum_log_abs_det_jacobians, dim=1,
                                                                                     keepdim=True).mean() / model.NI
    for idx, lj in enumerate(LogJoint):
        optimizer.zero_grad()
        if idx != len(LogJoint) - 1:
            lj.backward(retain_graph=True)
        else:
            lj.backward(retain_graph=False)

        if args.noisy == 'True':
            nn.utils.clip_grad_norm_(nf.parameters(), max_norm=args.C, norm_type=2)
        for i, p in enumerate(nf.parameters()):
            grads[i] += p.grad.clone()

    optimizer.zero_grad()
    if args.noisy == 'True':
        for i, p in enumerate(nf.parameters()):
            p.grad = grads[i] + (torch.randn(p.grad.size()) * args.C * args.sigma).to(device)
    else:
        for i, p in enumerate(nf.parameters()):
            p.grad = grads[i]
    optimizer.step()

    loss = LogJoint.sum()
    print("Iteration: {}\tLoss: {}".format(iteration, loss), end='\r')
    if iteration % args.log_interval == 0:
        print("Iteration: {}\tLoss: {}".format(iteration, loss))
        log.append([iteration, loss.item()])
