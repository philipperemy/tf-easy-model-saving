# Tensorflow Easy Model Saving
An easy way to load and save checkpoints in Tensorflow!

## Installation
```
git clone https://github.com/philipperemy/tf-easy-model-saving.git
cd tf-easy-model-saving
pip3 install .
```

## Usage

Check `example.py` for more information.

### Restore model variables  (<-)

```
from easy_model_saving import model_saver

# define graph and session

last_step = model_saver.restore_graph_variables(checkpoint_dir)

if last_step == 0:

    print('Did not find any weights.')
    model_saver.initialize_graph_variables()
    
else:

    print('Restore successful.')
```

### Save model variables (->)

```
from easy_model_saving import model_saver

# define graph and session

saver = model_saver.Saver(checkpoint_dir)

saver.save(global_step=1)  # global step is your epoch/gradient step.
```
