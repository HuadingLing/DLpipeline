"""
Deep Learning Pipeline for pytorch.
It is custom-made for the purpose of my own personal use.
However, if you find it helpful for your problems, feel free to modifie it to fit your problem, to serve as your tool.
I hope it can inspire your work.

Author: Huading LING, 2020.09.24
Project website: https://github.com/HuadingLing/DLpipeline
"""

"""
Basic framework:
    DLpipeline(Saver, Executor, Reporter, ProgressBar, FileNameManager, 
               PytorchThings(model, optim, dataloader, etc), other)
               
DLpipeline:
    Core class, connects every thing together
    Manage pipeline saving and loading
    Summary the pipeline
    Execute (train, val, test)
    Report the result
    Etc.
Saver:
    Everything about saving (to a file) and loading (from a file)
Executor:
    Handle the training and validating, testing process
Reporter: 
    Maintain a history dict, record anything you want
    Summary the pipeline
    Report the result of training (during training of after training), testing
    Save report fig
ProgressBar:
    Real-time display progress bar during training and testing, inculed time, loss, metric info
FileNameManager:
    Naming mechanism
    Mainly used by saver
"""

import sys
import time
from math import floor
import os
import re
import json
import torch
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

import itertools
import seaborn as sns
#sns.set_theme(palette='deep')
#sns.set_theme(palette="light:#5A9")


'''
Carefully use, might be some bugs
Try not to use OnelineBar, it's not under development


Next work：
    hist_sep_store (test and debug) 
    reporter (maybe need to further improve)
    label
    where 'pass' appear
    saver.update_meta
    further optimize
    parallel and distribute design (this version only consider about single/no GPU)

Problem:
    ProgressBar set dynamic to true does not work as expected (in interactive mode), 
    weird thing occurs and I don't know how to fix it.
'''


def output_to_score_fun(output):
    """ use softmax to normalize the unnormalize output of a DNN """
    return scipy.special.softmax(output, axis=1)

def reconstruct_from_cm(cm, labels = None):
    """
    when only save confusion matrix without output, use this function to
    reconstruct y_true and y_pred for the convenience of using classification_report of sklearn
    """
    y_true = []
    y_pred = []
    l = len(cm)
    if labels == None:
        labels = [i for i in range(l)]
    for i in range(l):
        for j in range(l):
            y_true.extend([labels[i] for _ in range(cm[i][j])])
            y_pred.extend([labels[j] for _ in range(cm[i][j])])
    return y_true, y_pred


def mkdir(path):
    """ check whether the path exist, if not, create it """
    if path and not os.path.exists(path):
        os.makedirs(path)
        

def get_parameter_number(net):
    """ get parameters of a pytorch model """
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable


def autolabel(ax, rects):
    """ Attach a text label above each bar in *rects*, displaying its height """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_confusion_matrix(cm, *, title='', labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          axes_style='dark',
                          context_mode = 'notebook', font_scale=1.0,
                          xticks_rotation='horizontal',
                          values_format=None, cmap='GnBu', 
                          ax=None, figsize=(10, 8), inverted_y=False, 
                          save_dir=None, **kwargs):
    """Plot Confusion Matrix.

    Parameters
    ----------
    cm: Confusion Matrix
    title: title of the figure
    labels : array-like of shape (n_classes,)
    sample_weight : array-like of shape (n_samples,)
    normalize : {'true', 'pred', 'all'}
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    display_labels : array-like of shape (n_classes,)
        Target names used for plotting. By default, `labels` will be used if
        it is defined.
    include_values : bool. Includes values in confusion matrix.
    axes_style: style of axes, see sns.axes_style
    context_mode: mode of context, see sns.plotting_context
    font_scale: scale of context, see sns.plotting_context
    xticks_rotation : {'vertical', 'horizontal'} or float. Rotation of xtick labels.
    values_format : str, Format specification for values in confusion matrix. If `None`,
        the format specification is 'd' or '.2g' whichever is shorter.
    cmap : str or matplotlib Colormap. Colormap recognized by matplotlib.
    ax : matplotlib Axes, Axes object to plot on. If `None`, a new figure and axes is created.
    figsize: tuple, size of the figure
    inverted_y: bool, whether to inverted y axis
    save_dir: str, path to save the figure. if set to None, the figure will not be saved.
    **kwargs: arguments of fig.savefig()
    ......

    """
    
    if normalize == 'true':
        cm /= np.sum(cm, axis=1)
    elif normalize == 'pred':
        cm /= np.sum(cm, axis=0)
    elif normalize == 'all':
        cm /= np.sum(cm)
        
    n_classes = cm.shape[0]
    if labels is None:
        labels = np.arange(n_classes)
    if display_labels is None:
        display_labels = labels
        
    with sns.axes_style(axes_style):
        with sns.plotting_context(context_mode, font_scale=font_scale):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, facecolor='white')
            else:
                fig = ax.figure

            im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            text_ = None
            cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

            if include_values:
                text_ = np.empty_like(cm, dtype=object)

                # print text with appropriate color depending on background
                thresh = (cm.max() + cm.min()) / 2.0

                for i, j in itertools.product(range(n_classes), range(n_classes)):
                    color = cmap_max if cm[i, j] < thresh else cmap_min

                    if values_format is None:
                        text_cm = format(cm[i, j], '.2g')
                        if cm.dtype.kind != 'f':
                            text_d = format(cm[i, j], 'd')
                            if len(text_d) < len(text_cm):
                                text_cm = text_d
                    else:
                        text_cm = format(cm[i, j], values_format)

                    text_[i, j] = ax.text(
                        j, i, text_cm,
                        ha="center", va="center",
                        color=color)

            fig.colorbar(im_, ax=ax)

            ax.set(xticks=np.arange(n_classes),
                   yticks=np.arange(n_classes),
                   xticklabels=display_labels,
                   yticklabels=display_labels,
                   ylabel="True label",
                   xlabel="Predicted label",
                   title=title)

            ax.set_ylim((n_classes - 0.5, -0.5))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
            if inverted_y:
                plt.gca().invert_yaxis()
            if save_dir is not None:
                fig.savefig(save_dir, **kwargs)
            plt.show()

    return fig, ax


class BasicExecutor():
    def __init__(self, pipeline = None):
        if pipeline:
            self.pipeline = pipeline
            pipeline.executor = self
        else:
            self.pipeline = None

    def train_prepare(self):
        pipeline = self.pipeline
        assert pipeline.optimizer is not None
        
        pipeline.saver.train_prepare()
        pipeline.reporter.train_prepare()
        pipeline.progressbar.train_prepare()
    
    def test_prepare(self):
        pipeline = self.pipeline
        pipeline.reporter.test_prepare()
        pipeline.progressbar.test_prepare()

    def __call__(self, end_epoch = 1, start_epoch = 1):
        pipeline = self.pipeline
        assert pipeline.model is not None
        pipeline.start_epoch = start_epoch
        pipeline.end_epoch = end_epoch
        
        reporter = pipeline.reporter
        saver = pipeline.saver
        history = reporter.history

        if pipeline.trainloader:
            self.train_prepare()

            for pipeline.epoch in range(start_epoch, end_epoch + 1):
                epoch_train_hist, reporter.train_metric = self.train()
                history['train'].append(epoch_train_hist)
                if pipeline.valloader:
                    epoch_val_hist, reporter.val_metric = self.validation()
                    history['val'].append(epoch_val_hist)
                reporter.check_and_report()  # report before save，防止清空 hist
                saver.check_and_save()
                
        else:
            print('No train set.')

        if pipeline.testloader:
            self.test_prepare()
            test_hist, reporter.test_metric = self.test()
            history['test'] = test_hist

        history['previous'] = saver.pre_file
        history['epoch'] = end_epoch

        if saver.save_history:
            saver.save_hist()

        reporter()
        saver.update_meta()
        
        
        pipeline.start_epoch = pipeline.epoch + 1 # update starting point

        return history


class Executor(BasicExecutor):
    def __init__(self, **kwargs):
        super(Executor, self).__init__(**kwargs)

    def train(self):
        pipeline = self.pipeline
        device = pipeline.device
        model = pipeline.model
        criterion = pipeline.criterion
        optimizer = pipeline.optimizer
        progressbar = pipeline.progressbar
        
        model.train()
        tot_loss = tot_correct = total = 0
        hist = []
        batch_size = pipeline.trainloader.batch_size
        format_display2 = FormatDisplay2(len(pipeline.trainloader.dataset))

        progressbar.bar_prepare('train', _format=format_display2._format)
        for batch_idx, (inputs, targets) in enumerate(pipeline.trainloader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            tot_loss += batch_loss
            num = targets.size(0)
            total += num
            with torch.no_grad():
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
            tot_correct += correct

            progressbar(batch_idx, (tot_loss / batch_idx if num == batch_size  # 'else' is for last batch correction， calculation is kind of weird, to prevent overflow
                                    else (tot_loss - batch_loss * (1 - num / batch_size)) * (batch_size / len(pipeline.trainloader.dataset)),
                                    tot_correct / total,
                                    format_display2(tot_correct, total),
                                    batch_loss,
                                    correct / num,
                                    correct,
                                    num))
            hist.append((batch_loss, correct))

        return hist, tot_correct / total

    def validation(self):
        pipeline = self.pipeline
        device = pipeline.device
        model = pipeline.model
        criterion = pipeline.criterion
        progressbar = pipeline.progressbar
        
        model.eval()
        tot_loss = tot_correct = total = 0
        batch_size = pipeline.valloader.batch_size
        format_display2 = FormatDisplay2(len(pipeline.valloader.dataset))

        progressbar.bar_prepare('val', _format=format_display2._format)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pipeline.valloader, 1):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                tot_loss += loss.item()
                num = targets.size(0)
                total += num
                _, predicted = outputs.max(1)
                tot_correct += predicted.eq(targets).sum().item()
                val_loss = tot_loss / batch_idx if num == batch_size \
                            else (tot_loss - loss.item() * (1 - num / batch_size)) * (batch_size / len(pipeline.valloader.dataset))

                progressbar(batch_idx, (val_loss,
                                        tot_correct / total,
                                        format_display2(tot_correct, total)))

        return (val_loss, tot_correct), tot_correct / total

    def test(self):
        pipeline = self.pipeline
        device = pipeline.device
        model = pipeline.model
        criterion = pipeline.criterion
        progressbar = pipeline.progressbar
        need_cm = pipeline.reporter.need_confusion_matrix
        need_output = pipeline.reporter.need_output
        need_store = need_cm or need_output
        
        
        if need_store:
            y_true = []
            if need_cm:
                y_pred = []
            if need_output:
                y_output = []
        #labels = self.pipeline.reporter.labels
        #cm = np.zeros((len(labels),len(labels)))
        
        model.eval()
        tot_loss = tot_correct = total = 0
        batch_size = pipeline.testloader.batch_size

        progressbar.bar_prepare('test')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pipeline.testloader, 1):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                tot_loss += loss.item()
                num = targets.size(0)
                total += num
                _, predicted = outputs.max(1)
                tot_correct += predicted.eq(targets).sum().item()
                test_loss = tot_loss / batch_idx if num == batch_size \
                            else (tot_loss - loss.item() * (1 - num / batch_size)) * (batch_size / len(pipeline.testloader.dataset))

                progressbar(batch_idx, (test_loss,
                                        tot_correct / total,
                                        tot_correct,
                                        total))
                # precision，recall，F1 can be considered
                
                if need_store:
                    y_true.extend(list(targets.cpu().numpy()))
                    if need_cm:
                        y_pred.extend(list(predicted.cpu().numpy()))
                    if need_output:
                        y_output.extend(list(outputs.cpu().numpy()))
                
        test_hist = {'test loss': test_loss}
        if need_cm:
            test_hist['confusion matrix'] = confusion_matrix(y_true, 
                                                             y_pred,
                                                             labels = self.pipeline.reporter.labels)
        if need_output:
            test_hist['y_true'] = y_true
            test_hist['output'] = y_output

        return test_hist, tot_correct / total


class BasicReporter():
    def __init__(self, 
                 pipeline = None,
                 init_hist = {},  # initialize history
                 report_interval = 0,  # report_interval>0 is the basic of showing and saving of the report
                 show_train_report = True,  # whether to print the report during training
                 summary_fun = None  # function to print the summary of current pipeline (model, optim, etc.)
                ):
        
        if pipeline:
            self.pipeline = pipeline
            pipeline.reporter = self
        else:
            self.pipeline = None
        self.history = init_hist
        self.report_interval = report_interval
        self.show_train_report = show_train_report
        self.summary_fun = summary_fun

    def clean_hist(self):
        self.history = {}
        
    def load_hist(self, hist):
        self.history = hist
        
    def sep_store_clean(self):
        self.history['tarin'] = []
        self.history['val'] = [] # val need to free???
        # if want to check overfit, is it better to retain val? 
        # and it does not take so much memory (not the same level as train)
        
    def set_summary(self, summary_fun = None):
        self.summary_fun = summary_fun

    def summary(self, **kwargs):
        pipeline = self.pipeline
        if self.summary_fun:
            self.summary_fun(pipeline, **kwargs)
        else:
            print('Model name: %s' % pipeline.model_name)
            print('Model arch:\n', pipeline.model)
            total, trainable = get_parameter_number(pipeline.model)
            print('Total model param: %d' % total)
            print('Trainable param: %d\n' % trainable)
            print('Optimizer:\n', pipeline.optimizer)
            
    def execute_prepare(self, init_hist = None):
        self.clean_hist()
        self.history['start epoch'] = self.pipeline.epoch if init_hist is None else init_hist['start epoch']
        #self.history['epoch'] = self.pipeline.epoch if init_hist is None else init_hist['epoch']
        self.history['hist_sep_store'] = self.pipeline.saver.hist_sep_store
        
        if init_hist and 'metric' in init_hist.keys() and init_hist['metric'] is not None:
            self.best_val_metric = self.val_metric = init_hist['metric']
        else:
            self.best_val_metric = self.val_metric = 0
        flag = True
        if self.history['hist_sep_store'] == False and init_hist is not None:
            if init_hist['hist_sep_store']:  # hist incomplete, need to load all the pieces
                init_hist, flag = self.pipeline.saver.load_previous_hist(init_hist, self.execute_save_dir)
            self.history['train'] = init_hist['train']
            self.history['val'] = init_hist['val']
        else:
            self.history['train'] = []
            self.history['val'] = []
            
        return flag

    def train_prepare(self):
        self.history['trainset size'] = len(self.pipeline.trainloader.dataset)
        self.history['batch size'] = self.pipeline.trainloader.batch_size
        if self.pipeline.valloader:
            self.history['valset size'] = len(self.pipeline.valloader.dataset)
        
    def test_prepare(self):
        self.history['testset size'] = len(self.pipeline.testloader.dataset)
        
        
class Reporter(BasicReporter):
    def __init__(self, 
                 labels = None,  # numerical value label of the classes
                 class_names = None,  # string, use to display the classes
                 need_output = True,  # whether to store the output of model during testing. useful for ROC and AUC
                 need_confusion_matrix = False, # whether to store confusion matrix during testing
                 output_to_score_fun = None,  # function to transfer the output of model to score-like value (i.e., sofmax, normalized, sum is 1)
                 batch_figsize = (14, 6),  # figsize of batch-loss-acc figure in the report
                 epoch_figsize = (10, 6), # figsize of epoch-loss-acc figure in the report
                 cm_figsize = (10, 8),  # figsize of confusion matrix figure in the report
                 cr_figsize = (18, 8),  # figsize of (sklearn's) classification_report figure in the report
                 roc_figsize = (10, 8),  # figsize of ROC curve figure in the report
                 **kwargs):
        super(Reporter, self).__init__(**kwargs)
        
        self.labels = labels
        if class_names is None:
            if labels is not None:
                self.class_names = [str(label) for label in labels]
            else:
                print('Error for the labels.')
        else:
            self.class_names = class_names
        assert len(self.labels) == len(self.class_names)
        self.need_output = need_output
        self.need_confusion_matrix = need_confusion_matrix
        self.output_to_score_fun = output_to_score_fun
        self.batch_figsize = batch_figsize
        self.epoch_figsize = epoch_figsize
        self.cm_figsize = cm_figsize
        self.cr_figsize = cr_figsize
        self.roc_figsize = roc_figsize
        
        
    def check_and_report(self):
        if self.report_interval > 0 and (self.pipeline.epoch - self.pipeline.start_epoch + 1) % self.report_interval == 0:
            self.history['epoch'] = self.pipeline.epoch
            if self.show_train_report or self.pipeline.save_train_report:
                self.plot_train(hist = self.history, in_train = True)
        
    def __call__(self, **kwargs):
        self.plot_hist(self.history, **kwargs)
        pass
    
    def plot_hist(self, hist, modes = 'all', **kwargs):
        if isinstance(modes, str):
            modes = [modes]
        assert isinstance(modes, list)
        if ('train' in modes or 'all' in modes) and 'train' in hist.keys() and hist['train']:
            self.plot_train(hist, 'val' in modes or 'all' in modes, **kwargs)
        if ('test' in modes or 'all' in modes) and 'test' in hist.keys() and hist['test']:
            self.plot_test(hist)

    def plot_train(self, hist, plot_val = True, in_train = False, drop_epochs = 0):
        start_epoch = hist['start epoch']
        temp_epoch = hist['epoch']
        e = range(start_epoch, temp_epoch + 1)
        batch_size = hist['batch size']
        train_size = hist['trainset size']
        last_size = train_size % batch_size  # size of the last batch
        iters = train_size // batch_size
        train_batch_loss = [l2[0] for l1 in hist['train'] for l2 in l1]
        train_batch_correct = [l2[1] for l1 in hist['train'] for l2 in l1]
        # train_loss = [l2[0]*batch_size for l1 in hist['train'] for l2 in l1]

        if iters == len(hist['train'][0]):  # len(hist['train'][0]) indicates how many train_batch in one epoch
            # no remnant batch
            train_batch_acc = [l2[1]/batch_size for l1 in hist['train'] for l2 in l1]

            train_loss = np.array(train_batch_loss).reshape(-1, iters)
            train_acc = np.array(train_batch_acc).reshape(-1, iters)
            train_loss = np.mean(train_loss, axis=1)
            train_acc = np.mean(train_acc, axis=1)

        else:  # the last train_batch is remnant
            train_batch_acc = [(l1[i][1]/batch_size if i < iters else l1[i][1]/last_size)
                               for l1 in hist['train'] for i in range(iters+1)]

            train_loss = np.array(train_batch_loss).reshape(-1, iters+1)
            train_acc = np.array(train_batch_correct).reshape(-1, iters+1)

            train_loss = (np.sum(train_loss[:,:-1], axis=1) * batch_size + train_loss[:,-1] * last_size) / train_size
            train_acc = np.sum(train_acc, axis=1) / train_size
        
        iters = range(1, len(train_batch_loss)+1)
        if drop_epochs > 0:
            if drop_epochs > len(hist['train']):
                drop_epochs = len(hist['train'])
            drop_iters = drop_epochs*len(hist['train'][0])
            iters = iters[drop_iters:]
            train_batch_loss = train_batch_loss[drop_iters:]
            train_batch_acc = train_batch_acc[drop_iters:]
            
        fig, ax1 = plt.subplots(figsize=self.batch_figsize, facecolor='white')
        color = 'C0'
        ax1.set_xlabel('Iters')
        ax1.set_ylabel('Batch Loss', color=color)
        ax1.scatter(iters, train_batch_loss, s=8, color=color, marker='2', label='batch loss')  
        # marker: ascent uses '1', descent used '2', and more like '.', '+', 'x' and '|' ,just personal preference
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'C2'
        ax2.set_ylabel('Batch Acc', color=color)  # we already handled the x-label with ax1
        ax2.scatter(iters, train_batch_acc, s=8, color=color, marker='1', label='batch acc')
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.legend(loc='upper left')
        ax2.legend(loc='lower left')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if self.pipeline.saver.save_train_report or (not in_train and self.pipeline.saver.save_test_report):
            # not in train equal to in test, depend on saver.save_test_report at this time
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report train'))
            fig.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        if not in_train or self.show_train_report:
            plt.show()



        fig, ax1 = plt.subplots(figsize=self.epoch_figsize, facecolor='white')
        plt.grid(True, which='major', axis='y')  # place grid first, to prevent it cover the line follow-up (put grid in the bottom layer)
        color = 'C0'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(e, train_loss, linestyle='-.', color=color, label='train loss')
        ax1.tick_params(axis='y', labelcolor=color)


        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'C2'
        ax2.set_ylabel('Acc', color=color)  # we already handled the x-label with ax1
        ax2.plot(e, train_acc, linestyle='-.', color=color, label='train acc')
        ax2.tick_params(axis='y', labelcolor=color)


        if plot_val and 'val' in hist.keys() and hist['val']:
            val_loss = [l[0] for l in hist['val']]
            val_acc = [l[1]/hist['valset size'] for l in hist['val']]
            if len(val_loss) == len(e):
                ev = e
            else:  # val incomplete, maybe some experiments haven't setup valloader
                ev = range(temp_epoch - len(val_loss) + 1, temp_epoch + 1)

            ax1.plot(e, val_loss, color='C4', label='val loss')
            ax2.plot(e, val_acc, color='C1', label='val acc')

        ax1.legend(loc='upper left')
        ax2.legend(loc='lower left')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if self.pipeline.saver.save_train_report or (not in_train and self.pipeline.saver.save_test_report):
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report val'))
            fig.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        if not in_train or self.show_train_report:
            plt.show()
        
    def plot_test(self, hist):
        test_dict = hist['test']
        y_true = None
        y_score = None
        y_pred = None
        cm = None
        
        # first see which we can get: y_true, y_score, y_pred, cm
        # use y_score can get y_pred
        # use y_true and y_pred can get cm
        # use cm can get y_true and y_pred
        if 'confusion matrix' in test_dict.keys():
            cm = test_dict['confusion matrix']
            # tn, fp, fn, tp = cm.ravel()
        if 'y_true' in test_dict.keys() and 'output' in test_dict.keys():
            y_true = test_dict['y_true']
            if self.output_to_score_fun:
                y_score = self.output_to_score_fun(test_dict['output'])
            else:
                y_score = np.array(test_dict['output'])  # make sure it's ndarray
            y_pred = np.argmax(y_score, axis=1)
            if cm is None:
                cm = confusion_matrix(y_true, y_pred, labels = self.labels)

        if cm is not None:
            assert len(cm) == len(self.labels)
            if self.pipeline.saver.save_test_report:
                save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report cm'))
            else:
                save_dir = None
            plot_confusion_matrix(cm, labels = self.labels, figsize=self.cm_figsize, 
                                  title = 'Confusion matrix',
                                  cmap='BuGn',  # can also use cmap='GnBu', just personal preference
                                  #xticks_rotation = 30,
                                  axes_style='dark', context_mode='notebook',
                                  save_dir = save_dir, transparent=False, dpi=80, bbox_inches="tight") 
            print('Confusion matrix:')
            print(cm)  # display numerical value (text)

            if y_true is None or y_pred is None:
                y_true, y_pred = reconstruct_from_cm(cm)
            cr_dict = classification_report(y_true, y_pred, labels = self.labels, target_names = self.class_names, digits = 4, output_dict = True)
            self.plot_classification_report(cr_dict, self.cr_figsize)
            cr = classification_report(y_true, y_pred, labels = self.labels, target_names = self.class_names, digits = 4, output_dict = False)
            print('\nClassification report:\n')
            print(cr)  # display classification report (text)
            print()
        if y_true is not None and y_score is not None:
            self.plot_roc(y_true, y_score)
            
    def plot_classification_report(self, cr, figsize=(16, 8), width=0.24):
        classes_labels = list(self.class_names)
        classes_labels.append('macro avg')
        classes_labels.append('weighted avg')
        precision = [cr[cl]['precision'] for cl in classes_labels]
        recall = [cr[cl]['recall'] for cl in classes_labels]
        f1 = [cr[cl]['f1-score'] for cl in classes_labels]

        classes_x = np.arange(len(classes_labels))  # the label locations

        fig, ax = plt.subplots(figsize=figsize, facecolor='w')
        ax.grid(True, which='major', axis='y')

        rects_precision = ax.bar(classes_x - width, precision, width, label='precision')
        rects_recall = ax.bar(classes_x, recall, width, label='recall')
        rects_f1 = ax.bar(classes_x + width, f1, width, label='f1-score')
        rects_acc = ax.bar(len(classes_x) - width, cr['accuracy'], width*1.2, color='C7', label='Acc')

        xticks = [i for i in classes_x]
        xticks.append(len(classes_x)-width)
        xticklabels = [cl+'\n('+str(cr[cl]['support']) + ')' for cl in classes_labels]
        xticklabels.append('Acc\n({:})'.format(cr['macro avg']['support']))

        ax.set_ylabel('Scores')
        ax.set_ylim([0.0, 1.02])
        ax.set_title('Scores by Multi-class')
        ax.set_xlabel('Items and support')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim([-3*width, len(classes_x)+width])
        ax.legend()

        autolabel(ax, rects_precision)
        autolabel(ax, rects_recall)
        autolabel(ax, rects_f1)
        autolabel(ax, rects_acc)

        fig.tight_layout()
        if self.pipeline.saver.save_test_report:
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report cr'))
            fig.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        plt.show()
            
    def plot_roc(self, y_true, y_score):
        assert len(y_true) == len(y_score)
        assert len(self.class_names) == y_score.shape[1]
        macro_roc_auc_ovo = roc_auc_score(y_true, y_score, multi_class="ovo", average="macro")
        weighted_roc_auc_ovo = roc_auc_score(y_true, y_score, multi_class="ovo", average="weighted")
        macro_roc_auc_ovr = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        weighted_roc_auc_ovr = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
        print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} (weighted by prevalence)"
              .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
        print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} (weighted by prevalence)"
              .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

        fpr = {}
        tpr = {}
        roc_auc = {}
        y_true_one_hot = np.zeros_like(y_score)
        for i, j in enumerate(y_true):
            y_true_one_hot[i][j] = 1
        for i, j in enumerate(self.class_names):
            fpr[j], tpr[j], _ = roc_curve(y_true_one_hot[:,i], y_score[:,i])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in self.class_names]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in self.class_names:
            mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(self.class_names)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        xlim1, ylim1 = [-0.02, 1.0], [0.0, 1.02]
        xlim2, ylim2 = [-0.01, 0.5], [0.5, 1.01]
        if self.pipeline.saver.save_test_report:
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report roc'))
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim1, ylim=ylim1, save_dir = save_dir)
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim2, ylim=ylim2, save_dir = save_dir[:-4]+' enlarge' + save_dir[-4:])  # enlarge version
        else:
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim1, ylim=ylim1)
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim2, ylim=ylim2)  # enlarge version
        
    def plot_roc_curves(self, fpr, tpr, roc_auc, plot_all_classes=True, figsize=(12, 10), xlim=[-0.02, 1.0], ylim=[0.0, 1.02], save_dir=None):
        # Plot all ROC curves
        plt.figure(figsize=figsize, facecolor='white')
        plt.grid(True, which='major', axis='both')
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average (area = {0:0.4f})'.format(roc_auc["micro"]),
                 linestyle='-', linewidth=1.5)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average (area = {0:0.4f})'
                       ''.format(roc_auc["macro"]),
                 linestyle='-', linewidth=1.5)
        if plot_all_classes:
            for i in self.class_names:
                plt.plot(fpr[i], tpr[i], lw=1.0, linestyle='--',
                         label='class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='darkslategray', lw=1.0, linestyle='--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves of multi-class')
        plt.legend(loc="lower right")
        if save_dir is not None:
            plt.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        plt.show()


class FileNameManager():
    def __init__(self, pipeline = None, file_name_config = None):
        if pipeline:
            self.pipeline = pipeline
            pipeline.file_name_manager = self
        else:
            self.pipeline = None
        self.name_config = self.default_name_config()
        if file_name_config is not None:
            self.setup(file_name_config)
        # self.name_config can upgrad to a function, rather than a dict
    
    def default_name_config(self):
        name_config = {}
        for i in ['ckpt', 'val', 'final', 'final_val']:
            name_config[i] = {'prefix': '', 
                               'suffix': ' ' + i, 
                               'extension': '.pt'  # same as pytorch defult extension
                             } 
            
        for i in ['report train', 'report val', 'report cm', 'report cr', 'report roc']:
            name_config[i] = {'prefix': i, 
                               'suffix': '', 
                               'extension': '.png'  # personal preference
                             } 

        name_config['meta'] = {'prefix': '', 
                               'suffix': 'meta', 
                               'extension': '.json'  # use json for easy to read
                              }  

        name_config['history'] = {'prefix': 'history ', 
                                  'suffix': '', 
                                  'extension': '.hist'  # personal preference
                                 } 

        return name_config
    
    def setup(self, name_config):
        for i in name_config.keys():
            if i in ['meta', 'ckpt', 'val', 'final', 'final_val', 'history',
                     'report train', 'report val', 'report cm', 'report cr', 'report roc']:
                for j in name_config[i].keys():
                    if j in ['prefix', 'suffix', 'extension']:
                        self.name_config[i][j] = name_config[i][j]
        
    
    def __call__(self, mode):
        # meta and report we only need to save one file, so we don't need time and epoch info in the file name
        if mode in ['ckpt', 'val', 'final', 'final_val']:
            file_name = self.name_config[mode]['prefix'] + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ' ' + \
                        self.pipeline.model_name + ' epoch_' + str(self.pipeline.epoch) + self.name_config[mode]['suffix'] + self.name_config[mode]['extension']
        elif mode in ['report train', 'report val', 'report cm', 'report cr', 'report roc']:
            file_name = self.name_config[mode]['prefix'] + self.name_config[mode]['suffix'] + self.name_config[mode]['extension']
        elif mode == 'meta':
            file_name = self.name_config['meta']['prefix'] + self.name_config['meta']['suffix'] + self.name_config['meta']['extension']
        elif mode == 'history':
            file_name = self.name_config['history']['prefix'] + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + \
                        ' ' + self.pipeline.model_name + ' epoch_' + str(self.pipeline.reporter.history['epoch']) + \
                        self.name_config['history']['suffix'] + self.name_config['history']['extension']
        return file_name
    
    def get_ckpt_type(self, file_name):
        for i in ['meta', 'history', 'ckpt', 'val', 'final', 'final_val', 'report']:
            if re.search(i, file_name):
                return i
    
    def get_epoch_time(self, file_name):
        if self.get_ckpt_type(file_name) in ['meta', 'history', 'report']:
            return -1
        e = re.search("epoch_\d+", file_name)
        t = re.search("\d+_\d+", file_name)
        if e and t:
            return float(e.group()[6:]) + float(t.group()[2:8]) * 1e-6 + float(t.group()[-6:]) * 1e-12
        else:
            return -1
        
    def get_time(self, file_name):
        if self.get_ckpt_type(file_name) in ['meta', 'history']:
            return -1
        t = re.search("\d+_\d+", file_name)
        if t:
            return float(t.group()[2:8]) + float(t.group()[-6:]) * 1e-6
        else:
            return -1


class Saver():
    def __init__(self,
                 pipeline = None,
                 save_dir = None,  # major work space, but execute_save_dir is more important when user specify a path in pipeline.load or pipeline.create_pipeline
                 meta_state_fun = None,  # function to decide what info to store in a meta file
                 model_state_fun = None,  # function to decide what info to store in a model file
                 save_meta_file = None,  # whether to save meta file before training
                 save_ckpt_model = False,  # whether to save ckpt model file during training
                 save_val_model = False,  # whether to save val model (the one with better val_mertric) during training, val model is a special ckpt model
                 save_final_model = False,  # whether to save final model (at the end epoch)
                 save_val_optim = True,  # whether to save optimizer when save val model (default save optimizer for not-val ckpt model)
                 save_final_optim = False,  # whether to save optimizer when save final model
                 save_history = False,  # whether to save history
                 save_interval = 0,  # saving interval, how many epochs to perform per save. (save_train_report is controled by reporter.report_interval instead of this one)
                 test_model_use = 'final',  
                 save_train_report = False,  # whether to save report during training
                 save_test_report = False,  # whether to save report for testing
                 hist_sep_store = False,  # whether to seperate store history (piece by piece), (maybe useful for memory limited?)
                 hist_state_fun = None,  # function to decide what info to store in history in the file
                 subsequent_train_setup_fun = None  # function to do somethings after loading pipeline or model (i.e. freeze some layers of the model)
                 ):
        
        if pipeline:
            self.pipeline = pipeline
            pipeline.saver = self
        else:
            self.pipeline = None
            
        has_dir = True if save_dir else False
        has_save_interval = True if save_interval > 0 else False
        self.save_meta_file = has_dir and save_meta_file
        self.save_final_model = has_dir and save_final_model
        self.save_val_model = has_dir and has_save_interval and save_val_model
        self.save_ckpt_model = has_dir and has_save_interval and save_ckpt_model
        self.save_history = has_dir and save_history
        self.save_train_report = has_dir and save_train_report
        self.save_test_report = has_dir and save_test_report
        
        self.enable_save = self.save_meta_file or self.save_ckpt_model or self.save_val_model or self.save_final_model \
                            or self.save_history or self.save_train_report or self.save_test_report
        if self.enable_save:
            self.save_dir = os.path.normpath(save_dir)
            mkdir(self.save_dir)
        else:
            self.save_dir = None
            
        self.save_val_optim = self.save_val_model and save_val_optim
        self.save_final_optim = self.save_final_model and save_final_optim
        
        self.meta_state_fun = meta_state_fun
        self.model_state_fun = model_state_fun
        self.save_interval = save_interval
        self.test_model_use = test_model_use
        self.hist_sep_store = hist_sep_store
        self.hist_state_fun = hist_state_fun
        
        self.subsequent_train_setup_fun = subsequent_train_setup_fun

        if test_model_use == 'val':
            assert save_val_model

        '''
        Whether you seperate history to store, the completeness of history is depended on self.save_history.
        history incomplete, only and only if no saving and seperate store happen together.
        This problem has been completeness guaranteed in self.save_model, which is cleaning history after saving
        '''

    def execute_prepare(self, pre_file = None):
        if self.enable_save:
            if pre_file is None or pre_file == '':  # create new folder automatically
                self.execute_save_dir = os.path.join(self.save_dir, 
                                                     time.strftime("%Y%m%d_%H%M%S", time.localtime()) + \
                                                     ' ' + self.pipeline.model_name)
                mkdir(self.execute_save_dir)
                self.pre_file = None
            else:  
                # 3 situation: exist file, exist dir, non-exist dir
                if os.path.isfile(pre_file):  # file
                    self.execute_save_dir, pre_file = os.path.split(pre_file)
                    self.pre_file = pre_file
                else:  # exist dir or non-exist dir
                    self.execute_save_dir = pre_file
                    self.pre_file = None
                    mkdir(self.execute_save_dir)
        # pre_file is use for muiti-branch training, and hist_sep_store, and other benefit.
    
    def train_prepare(self):
        if self.save_meta_file:
            self.save_meta()


    def check_and_save(self):
        pipeline = self.pipeline
        epoch = pipeline.epoch
        reporter = pipeline.reporter
        save_final_val = False
        if self.save_interval > 0 and (epoch - pipeline.start_epoch + 1) % self.save_interval == 0:
            if self.save_val_model and reporter.val_metric > reporter.best_val_metric:
                # if self.valloader == None, then val_metric can't be updated, this if condition is impossible to satisfy
                # val > ckpt: prioritise save val, if val has been save, then would not save the corresponding ckpt
                # final > ckpt: same as above
                # val == final, if both condition satisfy, then save 'Final_val'
                reporter.best_val_metric = reporter.val_metric
                if epoch == pipeline.end_epoch and self.save_final_model:
                    self.pre_file = self.val_model_file = self.save_model('final_val',
                                                                          save_optim = self.save_val_optim or self.save_final_optim,
                                                                          metric = reporter.best_val_metric,
                                                                          )
                    save_final_val = True  # prevent saving the same model by 'final'
                else:
                    self.pre_file = self.val_model_file = self.save_model('val',
                                                                          save_optim = self.save_val_optim,
                                                                          metric = reporter.best_val_metric,
                                                                          )

            # guarantee not to conflict with final
            elif self.save_ckpt_model and (epoch != pipeline.end_epoch or not self.save_final_model):
                self.pre_file = self.save_model('ckpt',
                                                save_optim = True, # ckpt is mean to be continue traning, default saving optimizer
                                                metric = reporter.best_val_metric if pipeline.valloader else reporter.train_metric,
                                                )
        if epoch == pipeline.end_epoch and not save_final_val and self.save_final_model:
            self.pre_file = self.save_model('final',
                                            save_optim = self.save_final_optim,
                                            metric = reporter.val_metric if pipeline.valloader else reporter.train_metric, 
                                           )  # not best_val_acc but val_metric

    def save_meta(self):
        print('Saving meta ...', end='')
        if self.meta_state_fun is None:
            state = {'model name': self.pipeline.model_name,
                     'optimizer name': self.pipeline.optimizer.__class__.__name__,
                     }
        else:
            state = self.meta_state_fun(self.pipeline)
        file_name = self.pipeline.file_name_manager('meta')
        with open(os.path.join(self.execute_save_dir, file_name), 'w') as f:
            json.dump(state, f, indent=4, separators=(',', ': '))
        print(' Meta has been save to {%s}' % os.path.join(self.execute_save_dir, file_name))
        return file_name
        # meta can also save: hyper-parameter, model discription, code to construct model,
        # task discription, dataset discription, etc.
        
    def update_meta(self):
        pass
    
    def load_meta(self, file_name):
        with open(file_name, 'r') as load_f:
            meta = json.load(load_f)
        return meta

    def save_model(self, mode, save_optim = False, metric = None):
        print('Saving model ...', end='')
        pipeline = self.pipeline
        hist = pipeline.reporter.history
        epoch = pipeline.epoch
        if self.model_state_fun is None:
            state = {'param': pipeline.model.state_dict(),
                     'epoch': epoch,
                     'metric': metric,
                     'previous': self.pre_file}
        else:
            state = self.model_state_fun(pipeline)
        if save_optim:
            if 'optimizer' not in state.keys():
                state['optimizer'] = {'name': pipeline.optimizer.__class__.__name__,
                                      'state': pipeline.optimizer.state_dict()
                                      }
            if self.save_history and 'history' not in state.keys():
                hist['epoch'] = epoch
                hist['previous'] = self.pre_file
                state['history'] = hist  # save_optim is for continue training, therefore, also need hist (when save_history==True)
        file_name = pipeline.file_name_manager(mode)
        torch.save(state, os.path.join(self.execute_save_dir, file_name))

        if save_optim and self.save_history and hist['hist_sep_store']:
            pipeline.reporter.sep_store_clean()
        print(' Model has been save to {%s}' % os.path.join(self.execute_save_dir, file_name))
        return file_name

    def load_model(self, file_name):
        state = torch.load(file_name)
        self.pipeline.model.load_state_dict(state['param'])
        return state
    
    def subsequent_train_setup(self):
        if self.subsequent_train_setup_fun:
            self.subsequent_train_setup_fun(self.pipeline)

    def save_hist(self):
        print('Saving history ...', end='')
        hist = self.pipeline.reporter.history
        hist['epoch'] = self.pipeline.epoch
        if self.hist_state_fun:
            hist = self.hist_state_fun(self.pipeline)
        file_name = self.pipeline.file_name_manager('history')
        torch.save(hist, os.path.join(self.execute_save_dir, file_name))
        print(' History has been save to {%s}' % os.path.join(self.execute_save_dir, file_name))
        return file_name

    def load_hist(self, file):
        # if os.path.isfile(hist_file):  # file
        file_dir = os.path.dirname(file)
        hist = torch.load(file)
        if 'param' in hist.keys():  # it's a DLpipeline model file, model(-optim-(hist))
            if 'history' in hist.keys():
                hist = hist['history']
            else:
                print('File does not contains history.')
                return None, False

        if hist['hist_sep_store'] == False:
            return hist, True
        else:
            return self.load_previous_hist(hist, file_dir)
        
    def load_previous_hist(self, hist, file_dir):
        train = []
        val = []
        flag = True
        pre_file = hist['previous']
        while pre_file or pre_file != '':  # find all the pieces forward
            if not os.path.isfile(os.path.join(file_dir, pre_file)):
                break
            temp = torch.load(os.path.join(file_dir, pre_file))
            if 'history' in temp.keys():
                train.extend(temp['history']['train'])
                val.extend(temp['history']['val'])
                if temp['history']['hist_sep_store'] == False:
                    break  # found all the pieces
            pre_file = temp['previous']

        train, val = train[::-1], val[::-1]
        train.extend(hist['train'])
        val.extend(hist['val'])
        hist['train'], hist['val'] = train, val
        if len(hist['train']) != hist['epoch'] - hist['start epoch'] + 1:
            print('Error! Can not find all pieces of history.')
            flag = False
        return hist, flag


class DLpipeline():
    KEYS_1 = ['executor', 'progressbar', 'reporter', 'saver', 'file_name_manager']
    KEYS_2 = ['trainloader', 'valloader', 'testloader',
              'model', 'model_name', 'optimizer', 'criterion', 'device',]
    def __init__(self, 
                 executor = None,
                 progressbar = None,
                 reporter = None, 
                 saver = None, 
                 file_name_manager = None,
                 trainloader = None, 
                 valloader = None,
                 testloader = None,
                 model = None, 
                 model_name = None,
                 optimizer = None, 
                 criterion = None,
                 device = None
                ):
        
        self.executor = executor
        if executor is not None:
            executor.pipeline = self
        self.progressbar = progressbar
        if progressbar is not None:
            progressbar.pipeline = self
        self.reporter = reporter
        if reporter is not None:
            reporter.pipeline = self
        self.saver = saver
        if saver is not None:
            saver.pipeline = self
        self.file_name_manager = file_name_manager
        if file_name_manager is not None:
            file_name_manager.pipeline = self
            
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.check_model_name()
        
        self.start_epoch = 1  # Default set to 1, update after loading 'pipeline' (in create_pipeline) successfully. In executor.__call__ , update after training
        self.epoch = 1  
        self.end_epoch = 1  # Change in executor.__call__ (before training)
        # this 3 attributes define the state of a pipeline, especially start_epoch

    def setup(self, **kwargs):
        for arg in kwargs.keys():
            if arg in DLpipeline.KEYS_1:
                self.__dict__[arg] = kwargs[arg]
                kwargs[arg].pipeline = self
            if arg in DLpipeline.KEYS_2:
                self.__dict__[arg] = kwargs[arg]
        self.check_model_name()

    def check_model_name(self):
        if self.model and self.model_name is None:
            if 'name' in model.__dict__:
                self.model_name = model.name
            else:
                self.model_name = self.model.__class__.__name__
    
    def summary(self, **kwargs):
        self.reporter.summary(**kwargs)
    
    def report(self, **kwargs):
        self.reporter(**kwargs)
        
    def load(self, obj = 'pipeline', file_or_folder = None, **kwargs):
        if obj not in ['pipeline', 'model', 'history', 'meta']:
            print('Wrong parameter. obj can only be one of: pipeline, model, history, meta.')
        if file_or_folder is not None and file_or_folder != '' and os.path.exists(file_or_folder):
            file_or_folder = os.path.normpath(file_or_folder)
            if obj == 'pipeline':
                self.create_pipeline(ckpt_file_or_folder = file_or_folder, **kwargs)
            elif obj == 'model':
                assert os.path.isfile(file_or_folder)
                self.saver.load_model(file_or_folder, **kwargs)
                self.saver.execute_save_dir = os.path.dirname(file_or_folder)  # Load successfully, need to update saver.execute_save_dir
                print('Load model successfully.')
            elif obj == 'history':
                assert os.path.isfile(file_or_folder)
                hist, flag = self.saver.load_hist(file_or_folder)
                if flag or ('allow_incomplete_hist' in kwargs.keys() and kwargs['allow_incomplete_hist'] == True):
                    self.reporter.load_hist(hist, **kwargs)
                    self.saver.execute_save_dir = os.path.dirname(file_or_folder)
                    print('Load history successfully.')
                else:
                    print('Fail to load history.')
            else:
                if os.path.isdir(file_or_folder):
                    file_or_folder = os.path.join(file_or_folder, self.file_name_manager('meta'))
                try:
                    meta = self.saver.load_meta(file_or_folder, **kwargs)
                    self.saver.execute_save_dir = os.path.dirname(file_or_folder)
                    print('Load meta successfully.')
                    return meta
                except:
                    print('No meta file.')
        else:
            print('Not a file or a folder.')
        
    def create_pipeline(self, ckpt_file_or_folder = None, mode='epoch', **kwargs):
        assert self.model is not None
        if ckpt_file_or_folder is not None and ckpt_file_or_folder != '':
            ckpt_file_or_folder = os.path.normpath(ckpt_file_or_folder)

            if os.path.isfile(ckpt_file_or_folder):  # file
                print('Processing file ...')
                flag = self.subsequent_train_prepare(ckpt_file_or_folder, **kwargs)
                if flag:
                    self.start_epoch = self.epoch  # Load successfully, update starting point
                return

            elif os.path.isdir(ckpt_file_or_folder):  # folder
                print('Processing folder ...')
                if mode == 'epoch':
                    key_fun = self.file_name_manager.get_epoch_time  # epoch + time/1e12
                elif mode == 'time':
                    key_fun = self.file_name_manager.get_time
                elif mode == 'val metric':
                    key_fun = 1
                    pass
                else:
                    pass
                
                file_list = os.listdir(ckpt_file_or_folder)
                file_list.sort(key = key_fun, reverse = True)  # sort from big to small

                for f in file_list:
                    if os.path.isfile(os.path.join(ckpt_file_or_folder, f)) and key_fun(f) > 0:
                        print('Checking file {%s} ...' % os.path.join(ckpt_file_or_folder, f))
                        flag = self.subsequent_train_prepare(os.path.join(ckpt_file_or_folder, f), **kwargs)
                        if flag:
                            self.start_epoch = self.epoch
                            return

                print('No ckpt satisfies, next start a new training branch.')
                # because of the naming mechanism and the pre_file mechanism, 
                # we don't have to worry about different training branch in the same folder
                self.start_epoch = self.epoch = 1  # start at 1, pipeline.epoch is also set to 1, because reporter.execute_prepare() use it
                self.saver.execute_prepare(ckpt_file_or_folder)
                self.reporter.execute_prepare()
                return
            else:  # create appoint dir
                self.start_epoch = self.epoch = 1
                self.saver.execute_prepare(ckpt_file_or_folder)
                print('Create new folder {%s}' % self.saver.execute_save_dir)
                self.reporter.execute_prepare()
                return
        else:  # creater new folder (named by time)
            self.start_epoch = self.epoch = 1
            self.saver.execute_prepare()
            print('Create new folder {%s}' % self.saver.execute_save_dir)
            self.reporter.execute_prepare()
            return False
    
    def subsequent_train_prepare(self, ckpt_file, use_stored_optim = True, allow_incomplete_hist = False):
        state = torch.load(ckpt_file)
        if 'param' in state.keys():  # first guarantee
            if (not use_stored_optim or 'optimizer' in state.keys()) and (allow_incomplete_hist or 'history' in state.keys()):  # overall judge
                
                last_epoch = state['epoch']
                self.epoch = last_epoch + 1
                self.saver.execute_prepare(ckpt_file)
                
                if 'history' in state.keys():
                    flag = self.reporter.execute_prepare(state['history'])
                    if flag == False and not allow_incomplete_hist:
                        print('History incomplete.')
                        return False
                    print('Load history successfully.')
                else:
                    self.reporter.execute_prepare()
                
                self.model.load_state_dict(state['param'])
                print('Load model successfully.')
                
                
                if use_stored_optim and 'optimizer' in state.keys():  # load optimizer
                    if self.optimizer is None or self.optimizer.__class__.__name__ != state['optimizer']['name']:  # change optim class
                        self.optimizer = getattr(torch.optim, state['optimizer']['name'])(self.model.parameters(), 0.01)
                    self.optimizer.load_state_dict(state['optimizer']['state'])
                    print('Load optimizer successfully.')
                    
                print('From epoch %d by ckpt file {%s}' % (last_epoch, ckpt_file))

                self.saver.subsequent_train_setup()  # somethings to do after loading model (freeze some layers, etc.)
                return True
            else:
                print('No optimizer state dict or history in the file.')
                return False
        else:
            print('No model state dict in the file. Failed to load model.')
            return False
        
    def __call__(self, end_epoch = 1):
        if end_epoch == 0:  # only test
            if self.testloader:
                self.executor.test_prepare()
                test_hist, self.reporter.test_metric = self.executor.test()
                self.reporter.history['test'] = test_hist
                self.reporter.plot_hist(self.reporter.history, 'test')
            else:
                print('No test set!')
        else:
            if end_epoch >= self.start_epoch:
                self.executor(end_epoch, self.start_epoch)
            else:
                print('End epoch is less than pipeline\'s start epoch.')


class FormatDisplay():
    """ use for epoch or iter display in progress bar, the max epoch or iter has to be defined """
    def __init__(self, m, sep='/'):
        m = str(m)
        self._format = '%' + str(len(m)) + 'd' + sep + m
        self.len = 2*len(m) + len(sep)

    def __call__(self, x):
        return self._format % x


class FormatDisplay2():
    """ use for correct and total of acc display in progress bar, the max possiable total has to be defined """
    def __init__(self, m, left='(', right=')', sep='/'):
        self.left = left
        self.right = right
        self.sep = sep
        self.len = 2*len(str(m)) + len(left+right+sep)
        self._format = '%-' + str(self.len) + 's'

    def __call__(self, x, y):
        return self._format % (self.left + str(x) + self.sep + str(y) + self.right)


def bar_head_format(config):
    """ use for table header in TableProgbar """
    return ''.join([c if isinstance(c, str) else ' '*c for c in config])



def format_time(seconds):
    """ use for time display in progress bar """
    if seconds == 0.0:
        return '0ms'
    
    # I found that using list, list.append(str) and ''.joint(list) is faster than str+str+...+str
    f = []
    i = 0

    days = floor(seconds / 86400)
    if days > 0:
        f.append('%dD' % days)
        i += 1

    seconds -= days*86400
    hours = floor(seconds / 3600)
    if hours > 0 and i <= 2:
        f.append('%dh' % hours)
        i += 1

    if i > 1:
        return ''.join(f)
    else:
        seconds -= hours*3600
        minutes = floor(seconds / 60)
        if minutes > 0:
            f.append('%dm' % minutes)
            i += 1

    if i > 1:
        return ''.join(f)
    else:
        seconds -= minutes*60
        secondsf = floor(seconds)
        if secondsf > 0:
            f.append('%ds' % secondsf)
            i += 1

    if i > 1:
        return ''.join(f)
    else:
        seconds -= secondsf
        millis = int(seconds*1000)
        if millis > 0:
            f.append('%dms' % millis)

    return ''.join(f)


"""
progress bar is inspired from: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
"""
class TableProgbar():
    def __init__(self, pipeline = None, total_bar_length = 60, dynamic = False):
        if pipeline:
            self.pipeline = pipeline
            pipeline.progressbar = self
        else:
            self.pipeline = None
        self.term_width = 0
        self.dynamic = dynamic
        self.total_bar_length = total_bar_length
        self.total = 1
        self.last_time = None  # time.time()
        self.begin_time = None  # self.last_time
        self.left_msg = ''
        self.mid_msg = ''
        self.double = False

    def set_table_head(self, head):
        head = bar_head_format(head)
        print('-'*(len(head)))
        print(head)
        print('-'*(len(head)))

    def set_leftmsg(self, left_msg):
        self.left_msg = left_msg

    def set_rightmsg_format(self, rightmsg_format):
        self.rightmsg_format = rightmsg_format
        
    def set_table_splite_len(self, l):
        self.table_splite_len = l

    def begin(self, total, state='train', double=False):  # double=True use for train state follow by val
        self.state = state
        self.total = total
        self.format_display = FormatDisplay(total)
        self.cur_char = '-' if double else '='
        if state != 'val':
            self.term_width = 0
            self.mid_msg = ''
        self.last_time = time.time()
        self.begin_time = self.last_time

    def __call__(self, current, msg_v=None):
        if self.state == 'val':
            rest_char = '-'
        else:
            rest_char = '.'

        sys.stdout.write(self.left_msg)

        cur_len = max(0, int(self.total_bar_length * current / self.total) - 1)
        rest_len = self.total_bar_length - cur_len - 1
        arrow = '>' if current < self.total else self.cur_char
        sys.stdout.write('[' + self.cur_char*cur_len + arrow + rest_char*rest_len + ']')

        msg = self.table_msg(current, msg_v)
        sys.stdout.write(msg)

        msg_len = len(self.left_msg) + (self.total_bar_length + 2) + len(msg)
        if msg_len < self.term_width:
            sys.stdout.write(' '*(self.term_width-msg_len))
        elif msg_len > self.term_width:
            self.term_width = msg_len

        '''
        # Go back to the center of the bar.
        for i in range(self.term_width-int(self.total_bar_length/2)+2):
            sys.stdout.write('\b')
        '''
        if (not self.dynamic) or (current == self.total and self.cur_char == '='):
            sys.stdout.write('\n')
        else:
            sys.stdout.write('\r')
        sys.stdout.flush()


    def table_msg(self, current, msg_v):
        L = []
        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        #eta = (self.total-current) * tot_time / current
        # use moving average seems more accurete but more unstable.
        if current == 1:
            eta = (self.total - 1) * step_time
        else:
            eta = (self.total - current) * (self.last_step_time * 0.9 + step_time * 0.1)
        self.last_step_time = step_time

        if self.state != 'val':
            L.append(' ' + self.format_display(current))
            L.append('   %-24s' % (format_time(tot_time) + ' (' + format_time(step_time) + '/step)'))
            L.append('  %-8s' % format_time(eta))

            if current == self.total and self.cur_char == '-':
                self.mid_msg = ''.join(L)

            if msg_v:
                msg = self.rightmsg_format % msg_v
                L.append(msg)
                if current == self.total and self.cur_char == '-':
                    self.mid_msg += msg[:self.table_splite_len]
        else:
            L.append(self.mid_msg)
            if msg_v:
                L.append(self.rightmsg_format % msg_v)

            L.append('%-24s' % (format_time(tot_time) + ' (' + format_time(step_time) + '/step)'))
            L.append('  %-8s' % format_time(eta))

        return ''.join(L)


class Progbar(TableProgbar):
    def __init__(self, **kwargs):
        super(Progbar, self).__init__(**kwargs)
        
    def train_prepare(self):
        pipeline = self.pipeline
        if pipeline.valloader:
            print('Train on %d samples, validate on %d samples.' %
                  (pipeline.reporter.history['trainset size'], pipeline.reporter.history['valset size']), end= '')
        else:
            print('Train on %d samples.' % pipeline.reporter.history['trainset size'], end= '')
        print(' Start training ...')
        
        self.tabel_bar_head('train')
            
    def test_prepare(self):
        saver = self.pipeline.saver
        if saver.test_model_use == 'val' and saver.val_model_file:
            print('Reload model from %s' % saver.val_model_file)
            saver.load_model(os.path.join(saver.execute_save_dir, saver.val_model_file))
        print('Start testing, test on %d samples.' % len(self.pipeline.testloader.dataset))

        self.tabel_bar_head('test')
    
    def tabel_bar_head(self, state):
        pipeline = self.pipeline
        if state == 'train':
            if pipeline.valloader:
                # this look at bar_prepare('train') : [4, '%.4f', 3, '%.4f', 1, _format, 4, '%.4f', 3, '%.4f', 1, '(%d/%d)']
                self.set_table_splite_len(4 + 6 + 3 + 6 + 1 + FormatDisplay2(len(pipeline.trainloader.dataset)).len)
                head = ['Epoch', FormatDisplay(pipeline.end_epoch).len + 2 + self.total_bar_length + 2 + FormatDisplay(len(pipeline.trainloader)).len + 3 - len('Epoch'),
                        'Time used', 24 + 2 - len('Time used'),
                        'ETA', 8 + 4 - len('ETA'),
                        'Loss', 6 + 3 - len('Loss'),
                        'Acc', 6 + 1 +
                        FormatDisplay2(len(pipeline.trainloader.dataset)).len + 4 - len('Acc'),
                        'ValLoss', 6 + 3 - len('ValLoss'),
                        'ValAcc', 6 + 1 +
                        FormatDisplay2(len(pipeline.valloader.dataset)).len + 4 - len('ValAcc'),
                        'Time used', 24 + 2 - len('Time used'),
                        'ETA', 8 - len('ETA')]
            else:
                head = ['Epoch', FormatDisplay(pipeline.end_epoch).len + 2 + self.total_bar_length + 2 + FormatDisplay(len(pipeline.trainloader)).len + 3 - len('Epoch'),
                        'Time used', 24 + 2 - len('Time used'),
                        'ETA', 8 + 4 - len('ETA'),
                        'Loss', 6 + 3 - len('Loss'),
                        'Acc', 6 + 1 +
                        FormatDisplay2(len(pipeline.trainloader.dataset)).len + 4 - len('Acc'),
                        'BatLoss', 6 + 3 - len('BatLoss'),
                        'BatAcc', 6 + 1 + FormatDisplay2(pipeline.trainloader.batch_size).len - len('BatAcc')]

        else:
            head = [1 + self.total_bar_length + 2 + FormatDisplay(len(pipeline.testloader)).len + 3,
                    'Time used', 24 + 2 - len('Time used'),
                    'ETA', 8 + 4 - len('ETA'),
                    'Loss', 6 + 3 - len('Loss'),
                    'Acc', 6 + 1 + FormatDisplay2(len(pipeline.testloader.dataset)).len - len('Acc')]

        self.set_table_head(head)

    def bar_prepare(self, state, **kwargs):
        pipeline = self.pipeline
        if state == 'train':
            '''
            if self.mode == 'oneline':
                self.set_leftmsg('Epoch: ' + FormatDisplay(pipeline.end_epoch)(epoch) + ' ')
                self.set_rightmsg_format('loss: %.4f - acc: %.4f ' + _format + ' - batchloss: %.4f - batchacc: %.4f (%d/%d)')
            else:
            '''
            self.set_leftmsg(FormatDisplay(pipeline.end_epoch)(pipeline.epoch) + ' ')
            self.set_rightmsg_format(bar_head_format([4, '%.4f', 3, '%.4f', 1, kwargs['_format'], 4, '%.4f', 3, '%.4f', 1, '(%d/%d)']))
            self.begin(len(pipeline.trainloader), 'train', True if pipeline.valloader else False)

        elif state == 'val':
            '''
            if self.mode == 'oneline':
                self.set_rightmsg_format('val_loss: %.4f - val_acc: %.4f ' + _format)
            else:
            '''
            self.set_rightmsg_format(bar_head_format([4, '%.4f', 3, '%.4f', 1, kwargs['_format'], 4]))
            self.begin(len(pipeline.valloader), 'val')

        else:
            '''
            if self.mode == 'oneline':
                self.set_leftmsg('Test     ' + ' ' * (2*len('%d' % (pipeline.end_epoch))))
                self.set_rightmsg_format('test_loss: %.4f - test_acc: %.4f (%d/%d)')
            else:
            '''
            self.set_leftmsg('')
            self.set_rightmsg_format(bar_head_format([4, '%.4f', 3, '%.4f', 1, '(%d/%d)']))
            self.begin(len(pipeline.testloader), 'test')
            
'''
from time import sleep

def progress(percent=0, width=200):
    left = width * percent // 100
    right = width - left
    sys.stdout.write('[' + '#' * left + ' ' * right + ']' + f' {percent:.0f}%')
    sleep(0.001)
    sys.stdout.write('\r')
    
    sys.stdout.flush()

for i in range(101):
    progress(i)
    sleep(0.01)
'''


class OnelineProgbar():  # Try not to use this one, because it's not under development
    def __init__(self, pipeline = None, total_bar_length = 60, dynamic = False):
        if pipeline:
            self.pipeline = pipeline
            pipeline.progressbar = self
        else:
            self.pipeline = None
        self.term_width = 0
        self.dynamic = dynamic
        self.total_bar_length = total_bar_length
        self.total = 100
        self.last_time = None
        self.begin_time = None
        self.left_msg = ''
        self.mid_msg = ''
        self.double = False

    def set_leftmsg(self, left_msg):
        self.left_msg = left_msg

    def set_rightmsg_format(self, rightmsg_format):
        self.rightmsg_format = rightmsg_format
    
    def msg_split(self, msg):
        return msg.split('- b')[0]

    def begin(self, total, state='train', double=False):  # double=True use for train state follow by val
        self.state = state
        self.total = total
        self.format_display = FormatDisplay(total)
        self.cur_char = '-' if double else '='
        if state != 'val':
            self.term_width = 0
            self.mid_msg = ''
        self.last_time = time.time()
        self.begin_time = self.last_time

    def __call__(self, current, msg_v=None):
        if self.state == 'val':  # val
            rest_char = '-'
        else:
            rest_char = '.'

        sys.stdout.write(self.left_msg)

        cur_len = max(0, int(self.total_bar_length * current / self.total) - 1)
        rest_len = self.total_bar_length - cur_len - 1
        arrow = '>' if current < self.total else self.cur_char
        sys.stdout.write('[' + self.cur_char*cur_len + arrow + rest_char*rest_len + ']')

        msg = self.oneline_msg(current, msg_v)
        sys.stdout.write(msg)

        msg_len = len(self.left_msg) + (self.total_bar_length + 2) + len(msg)
        if msg_len < self.term_width:
            sys.stdout.write(' '*(self.term_width-msg_len))
        elif msg_len > self.term_width:
            self.term_width = msg_len

        '''
        # Go back to the center of the bar.
        for i in range(self.term_width-int(self.total_bar_length/2)+2):
            sys.stdout.write('\b')
        '''
        if (not self.dynamic) or (current == self.total and self.cur_char == '='):
            sys.stdout.write('\n')
        else:
            sys.stdout.write('\r')
        sys.stdout.flush()

    def oneline_msg(self, current, msg_v):
        L = []
        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        if self.state != 'val':
            L.append(' ' + self.format_display(current))

            L.append(' - tot: %-24s' % (format_time(tot_time) + ' (' + format_time(step_time) + '/step)'))
            if current < self.total:
                eta = (self.total-current) * tot_time / current
                L.append(' - ETA: %-8s' % format_time(eta))

            if current == self.total and self.cur_char == '-':
                self.mid_msg = ''.join(L)

            if msg_v:
                msg = self.rightmsg_format % msg_v
                L.append(' - ' + msg)
                if current == self.total and self.cur_char == '-':
                    self.mid_msg += ' - ' + self.msg_split(msg)

        else:
            L.append(self.mid_msg)
            if msg_v:
                L.append('- ' + self.rightmsg_format % msg_v)

            L.append(' - tot: %-s' % (format_time(tot_time) + ' (' + format_time(step_time) + '/step)'))
            if current < self.total:
                eta = (self.total-current) * tot_time / current
                L.append(' - ETA: %-8s' % format_time(eta))

        return ''.join(L)


