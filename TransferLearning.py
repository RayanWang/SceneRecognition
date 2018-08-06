import matplotlib.pyplot as plt
plt.switch_backend('agg')

from fastai.conv_learner import *
from fastai.dataset import *
from fastai.plots import *


arch = resnet50
bs = 16


def get_data(data_dir, sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(path=data_dir, tfms=tfms, bs=bs)

    return data


def plot_loss_change(sched, sma=1, n_skip=20, y_lim=(-0.01, 0.01)):
    """
    Plots rate of change of the loss function.
    Parameters:
        sched - learning rate scheduler, an instance of LR_Finder class.
        sma - number of batches for simple moving average to smooth out the curve.
        n_skip - number of batches to skip on the left.
        y_lim - limits for the y axis.
    """
    derivatives = [0] * (sma + 1)
    for i in range(1 + sma, len(sched.lrs)):
        derivative = (sched.losses[i] - sched.losses[i - sma]) / sma
        derivatives.append(derivative)

    plt.ylabel("d/loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(sched.lrs[n_skip:], derivatives[n_skip:])
    plt.xscale('log')
    plt.ylim(y_lim)


def train(data_dir):
    data = get_data(data_dir, 224, bs)
    learn = ConvLearner.pretrained(arch, data, ps=0.5, precompute=True)

    print('\nFirst training...\n')
    # learn.lr_find()
    # learn.sched.plot()
    learn.fit(1e-2, 1)

    # using data augmentation
    print('\nUsing data augmentation...\n')
    learn.precompute = False
    learn.fit(1e-2, 3, cycle_len=1)

    # fine tune model, re-train all weights in the convolutional kernels
    print('\nUnfreeze all layers...\n')
    learn.unfreeze()
    learn.bn_freeze(True)
    lrs = np.array([1e-5, 1e-4, 1e-2])
    learn.fit(lrs, n_cycle=3, cycle_len=1, cycle_mult=2)

    # Starting training on small images for a few epochs, then switching to bigger images,
    # and continuing training is an amazingly effective way to avoid overfitting.
    print('\nUsing bigger size with freezing for training...\n')
    learn.set_data(get_data(data_dir, 299, bs))
    learn.freeze()
    learn.fit(lrs, n_cycle=3, cycle_len=1, cycle_mult=2)

    print('\nUsing bigger size with unfreezing for training...\n')
    learn.unfreeze()
    learn.bn_freeze(True)
    learn.fit(lrs, n_cycle=3, cycle_len=1, cycle_mult=2)

    print('\nTest time augmentation...\n')
    log_preds, y = learn.TTA()
    probs = np.mean(np.exp(log_preds), 0)
    print('Accuracy: %.4f%%\n' % (accuracy_np(probs, y) * 100))


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print('Arguments must match:\npython code/TransferLearning.py <data_dir/>')
        sys.exit(2)
    else:
        data_dir = os.path.abspath(sys.argv[1])
        train_dir = os.path.join(os.path.abspath(data_dir), 'train')
        validate_dir = os.path.join(os.path.abspath(data_dir), 'validate')

    train(data_dir)
