import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import scipy.interpolate


class LearningRateTuner(object):
    """
    Learning Rate Tuner measures a model's training loss over a range of learning rates and suggests an optimal learning rate.
    The approach is based on Leslie Smith's Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    It has been adapted according to fastai's implementation: https://github.com/fastai/fastai

    Arguments:
        net (torch.nn.Module): required
        optimizer (torch.optim.Optimizer): required
        criterion (torch.nn.Module): required
        data_loader (torch.utils.data.DataLoader): required
    Example:
        >>> lr_tuner = LearningRateTuner(net=net, optimizer=optimizer, criterion=criterion, data_loader=data_loader)
        >>> optimal_learning_rate = lr_tuner.tune_learning_rate()
    """
    def __init__(self, **kwargs):
        if isinstance(kwargs.get('net'), torch.nn.Module):
            self.net = kwargs.get('net')
        else:
            raise TypeError('net must be of type torch.nn.Module')

        if isinstance(kwargs.get('optimizer'), torch.optim.Optimizer):
            self.optimizer = kwargs.get('optimizer')
        else:
            raise TypeError('net must be of type torch.optim.Optimizer')

        if isinstance(kwargs.get('criterion'), torch.nn.Module):
            self.criterion = kwargs.get('criterion')
        else:
            raise TypeError('net must be of type torch.nn.Module')

        if isinstance(kwargs.get('data_loader'), torch.utils.data.DataLoader):
            self.data_loader = kwargs.get('data_loader')
        else:
            raise TypeError('net must be of type torch.utils.data.DataLoader')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._init_tuning_params()

    def _init_tuning_params(self):
        """

        Set the learning rate tuning parameters

        """
        self.init_lr = 1e-8
        self.final_lr = 10
        self.beta = 0.98
        self.plot_smoothing_factor = 5
        self.plot_clip_start = 10
        self.plot_clip_end = -5
        return

    def tune_learning_rate(self):
        """

        Description:
            Iterate over the training set, adjusting the learning rate exponentially from 1e-8,...,10 for each batch.
            Record and plot the training loss for each batch.
            See Sylvain Gugger's post below for a thorough explanation of each calculation
                https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

        """
        self.net.to(self.device)
        tuning_steps = len(self.data_loader) - 1
        lr_multiple = (self.final_lr / self.init_lr) ** (1 / tuning_steps)
        lr = self.init_lr
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        training_losses = []
        lrs = []
        for batch_num, data in enumerate(self.data_loader):

            inputs, labels = self._extract_inputs_and_labels(data)
            # 1. Forward pass
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            # 2. Calculate Loss Variables
            model_loss = self.criterion(outputs, labels)
            avg_loss, best_loss, smoothed_loss = self._calculate_losses(model_loss, avg_loss, best_loss, batch_num)
            if batch_num > 1 and smoothed_loss > 4 * best_loss: break

            # 3. Backward propagation
            model_loss.backward()
            self.optimizer.step()

            # 4. Update tuning variables
            training_losses.append(smoothed_loss)
            lrs.append(lr)

            # 5. Set learning rate for next batch
            lr *= lr_multiple
            self.optimizer.param_groups[0]['lr'] = lr

        return self._plot_learning_rate_test(lrs=lrs, losses=training_losses)

    def _calculate_losses(self, model_loss, avg_loss, best_loss, batch_num):
        """
        Description:
            Calculate the average loss and best loss after each forward pass
            See Sylvain Gugger's post below for a thorough explanation of each calculation
                https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        Args:
            model_loss:
            avg_loss:
            best_loss:
            batch_num:

        Returns:

        """
        avg_loss = self.beta * avg_loss + (1 - self.beta) * model_loss.data
        smoothed_loss = avg_loss / (1 - self.beta**batch_num)
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        return avg_loss, best_loss, smoothed_loss

    def _extract_inputs_and_labels(self, data):
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        return Variable(inputs), Variable(labels)

    def _plot_learning_rate_test(self, lrs=None, losses=None):
        """
        Description:
            Plot the training losses for each learning rate used in tune_learning_rate()
            Function based on:
                https://github.com/fastai/fastai/blob/master/fastai/basic_train.py#L522

        Args:
            lrs:
            losses:

        Returns: Optimal learning rate

        """
        optimal_learning_rate = None
        lrs, losses = self._preprocess_plot_data(lrs=lrs, losses=losses)
        fig, ax = plt.subplots(1, 1)
        ax.plot(lrs, losses)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        try:
            min_loss_grad_idx = (np.gradient(np.array(losses))).argmin()
            optimal_learning_rate = lrs[min_loss_grad_idx]
            print('Min numerical gradient: {:.2E}'.format(optimal_learning_rate))
            ax.plot(lrs[min_loss_grad_idx], losses[min_loss_grad_idx], markersize=10, marker='o', color='red')
        except:
            print('Failed to compute the gradients, there might not be enough points.')
        plt.show()
        return optimal_learning_rate

    def _preprocess_plot_data(self, lrs=None, losses=None):
        """
        Description:
            Clip and smooth data prior to plotting
        Args:
            lrs:
            losses:

        Returns:

        """
        lrs = self._clip_plot_data(lrs)
        losses = self._clip_plot_data(losses)
        losses = self._smoothen_learning_rate_plot(losses)
        return lrs, losses

    def _clip_plot_data(self, data):
        return data[self.plot_clip_start: self.plot_clip_end]

    def _smoothen_learning_rate_plot(self, losses):
        """
        Description:
            Smooth the loss data prior to plotting:
                Makes the plot more legible
                Easier to extract the optimal learning rate
        Args:
            losses:

        Returns:
            Smoothed losses

        """
        x_series = np.arange(len(losses))
        spl = scipy.interpolate.UnivariateSpline(x_series, losses, s=self.plot_smoothing_factor)
        return spl(x_series)
