import logging
import os
import paddle
from collections import OrderedDict
from copy import deepcopy

from models import lr_scheduler as lr_scheduler

logger = logging.getLogger('basicsr')


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, prefix_name):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter, save_img=False, rgb2bgr=True, use_image=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
            rgb2bgr (bool): Whether to save images using rgb2bgr. Default: True
            use_image (bool): Whether to use saved images to compute metrics (PSNR, SSIM), if not, then use data directly from network' output. Default: True
        """

        return self.nondist_validation(dataloader, current_iter,
                                save_img, rgb2bgr, use_image)

    def get_current_log(self):
        return self.log_dict

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type == 'CosineAnnealingRestartCyclicLR':
            self.schedulers.append(
                lr_scheduler.CosineAnnealingRestartCyclicLR(**train_opt['scheduler']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        return net

    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """

        net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params.numpy()[0]:,d}')
        logger.info(net_str)


    def get_current_learning_rate(self):
        return [
            self.optimizers[0].get_lr()
        ]


    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with paddle.no_grad():
            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
