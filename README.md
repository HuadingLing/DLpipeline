# DLpipeline
Deep Learning Pipeline for pytorch.

It is custom-made for the purpose of my own personal use.

However, if you find it helpful for your problems, feel free to modifie it to fit your problem, to serve as your tool.

I hope it can inspire your work.

### Basic framework

DLpipeline(Saver, Executor, Reporter, ProgressBar, FileNameManager, PytorchThings(model, optim, dataloader, etc), other)

- DLpipeline:
- 1. Core class, connects every thing together
  2. Manage pipeline saving and loading
  3. Summary the pipeline
  4. Execute (train, val, test)
  5. Report the result
  6. Etc.
- Saver:
- 1. Everything about saving (to a file) and loading (from a file)
- Executor:
- 1. Handle the training and validating, testing process
- Reporter:
- 1. Maintain a history dict, record anything you want
  2. Summary the pipeline
  3. Report the result of training (during training of after training), testing
  4. Save report fig
- ProgressBar:
- 1. Real-time display progress bar during training and testing, inculed time, loss, metric info
- FileNameManager:
- 1. Naming mechanism
  2. Mainly used by saver

test.ipynb shows how to use it.

### Basic operation

You might have to modify *Reporter* and *Executor* to fit your problem. Then

```python
from dlpipeline import DLpipeline, Executor, Progbar, Reporter, Saver, FileNameManager
```

Use this to create a new pipeline, it will create a new folder under *saver.save\_dir*. Then every thing will be saved in this folder (if enable save). The name of the created folder is controlled by FileNameManager, default to time+model_name.
```python
pipeline.create_pipeline()
```

Use this to create a new pipeline, it will create a new folder with its path as user specify. Then every thing will save in this folder (if enable save). If the folder exists, then it will load the satify file with the max epoch. If not satify file, it will start a new training branch (two different pipeline share the same folder, try to avoid this because file cover will happen)
```python
pipeline.create_pipeline(save_dir + 'vgg_test/')
```

Use this to load an exist pipeline (mainly model+optim+hist+last_epoch).
```python
pipeline.load('pipeline', save_dir + 'vgg_test/20200923_225741 vgg_test epoch_18 val.pt')
```

Use this to load an exist pipeline, without specify file name, let the program to find the satify file with the max epoch. If not satify file, it will start a new training branch (two different pipelines share the same folder, try to avoid this because file cover will happen).
```python
pipeline.load('pipeline', save_dir + 'vgg_test/')
```

Use this to load a exist history, this history is not mean to serve as continue training. If you want to continue training, load pipeline or model instead of history.
```python
pipeline.load('history', save_dir + 'vgg_test/history 20200923_225937 vgg_test epoch_20.hist')
```

Use this to report the result (train and test).
```python
pipeline.report()
```

Use this to report the train result.
```python
pipeline.report(modes = ['train', 'val'])
```

Use this to report the test result.
``` python
pipeline.report(modes = 'test') 
```

The state of the pipeline (or *pipeline.start\_epoch*) only change when successfully load or create 'pipeline' using:

```python
pipeline.load('pipeline', ...)
# or
pipeline.create_pipeline(...)
```

More info, can be explore by using *\_\_dict\_\_*

```python
pipeline.__dict__
pipeline.saver.__dict__
pipeline.reporter.__dict__

history = pipeline.reporter.history
history.keys()
```