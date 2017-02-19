import os
import numpy as np

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams.update({'font.size': 15})

from nideep.eval.learning_curve import LearningCurve
from nideep.eval.eval_utils import Phase
import nideep.eval.log_utils as lu

from pycaffe.utils.output_grabber import correct_log

# List all logs in logs
logs = []
for (dirpath, dirnames, filenames) in os.walk('logs'):
    logs += filenames
    break

# Correct logs if needed

# Plot from logs
for filename in logs:
    log_path = 'logs/'+filename
    #correct_log(log_path)
    e = LearningCurve(log_path)
    e.parse()

    fig, ax1 = plt.subplots()
    for phase in [Phase.TRAIN, Phase.TEST]:
        num_iter = e.list('NumIters', phase)
        loss = e.list('loss_classification', phase)
        ax1.plot(num_iter, loss, label='loss on %s set' % (phase,))
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('loss')
        plt.title(e.name())
        ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    num_iter = e.list('NumIters', phase)
    acc = e.list('accuracy_classification', phase)
    ax2.plot(num_iter, acc, '-', color='r', label='accuracy')
    ax2.set_ylabel('accuracy')
    ax2.legend(loc='upper right')
    plt.grid()
    plt.savefig('figs/'+filename)
    plt.close(fig)
