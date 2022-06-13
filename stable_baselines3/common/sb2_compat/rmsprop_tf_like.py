from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer


class RMSpropTFLike(Optimizer):
    r"""Implements RMSprop algorithm with closer match to Tensorflow version.

    For reproducibility with original stable-baselines. Use this
    version with e.g. A2C for stabler learning than with the PyTorch
    RMSProp. Based on the PyTorch v1.5.0 implementation of RMSprop.

    See a more throughout conversion in pytorch-image-models repository:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/rmsprop_tf.py

    Changes to the original RMSprop:
        - Move epsilon inside square root
        - Initialize squared gradient to ones rather than zeros

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    :params: iterable of parameters to optimize or dicts defining
        parameter groups
    :param lr: learning rate (default: 1e-2)
    :param momentum: momentum factor (default: 0)
    :param alpha: smoothing constant (default: 0.99)
    :param eps: term added to the denominator to improve
        numerical stability (default: 1e-8)
    :param centered: if ``True``, compute the centered RMSProp,
        the gradient is normalized by an estimation of its variance
    :param weight_decay: weight decay (L2 penalty) (default: 0)

    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], None]] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.

        :param closure: A closure that reevaluates the model
            and returns the loss.
        :return: loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSpropTF does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # PyTorch initialized to zeros here
                    state["square_avg"] = torch.ones_like(p, memory_format=torch.preserve_format)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    # PyTorch added epsilon after square root
                    # avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add_(group["eps"]).sqrt_()
                else:
                    # PyTorch added epsilon after square root
                    # avg = square_avg.sqrt().add_(group['eps'])
                    avg = square_avg.add(group["eps"]).sqrt_()

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group["lr"])
                else:
                    p.addcdiv_(grad, avg, value=-group["lr"])

        return loss
