"""
A module for convolutions with cudnn.
"""

__author__ = "Nicolas Ballas"
__license__ = "3-clause BSD"
__credits__ = "Nicolas Ballas and Francesco Visin"
__maintainer__ = "Lisa Lab"

import functools
import numpy as np

from theano.sandbox.cuda.dnn import GpuDnnConv, GpuDnnConvDesc
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty

from pylearn2.packaged_dependencies.theano_linear.conv2d \
    import Conv2d as OrigConv2D

from pylearn2.linear.linear_transform import LinearTransform as P2LT
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng


default_seed = [2012, 11, 6, 9]
default_sparse_seed = [2012, 11, 6]


class Cudnn2D(OrigConv2D):
    """
    Wrapper on the Theano Cudnn op.

    Parameters
    ----------
    filters : Theano shared variable
        4D-tensor of shape (out channels, in channels, rows, cols)
    batch_size : int
        The size of the input batches
    input_space : Space
        The Space of the input data
    output_axes : tuple, optional
        The requested output axes. If not specified `bc01` will be used.
    subsample : tuple or list, optional
        Factor by which to subsample the output. Default (1, 1)
    border_mode : string, optional
        `valid` or `full`. See scipy.signal.convolve2d
    filters_shape : tuple of length 2 or 3, optional
        ([filter's number,] filter's height, filter's width)
    message : string, optional
        TODO
    """

    def __init__(self,
                 filters,
                 batch_size,
                 input_space,
                 output_axes=('b', 'c', 0, 1),
                 subsample=(1, 1),
                 border_mode='valid',
                 filters_shape=None,
                 message=''):

        assert batch_size is None or batch_size > 0
        self._input_space = input_space
        self._output_axes = output_axes
        self._subsample = tuple(subsample)
        self._border_mode = border_mode

        super(Cudnn2D, self).__init__(
            filters=filters,
            img_shape=(batch_size, input_space.num_channels,
                       input_space.shape[0], input_space.shape[1]),
            subsample=self._subsample,
            border_mode=border_mode,
            filters_shape=filters.get_value(borrow=True).shape,
            message=message
        )

        # conv_op has to be changed
        self._conv_op = GpuDnnConv()
        self._desc = GpuDnnConvDesc(border_mode=border_mode,
                                    subsample=self._subsample,
                                    conv_mode='conv')

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        """ Return self._filters. """
        return [self._filters]

    @functools.wraps(P2LT.get_weights_topo)
    def get_weights_topo(self, borrow):
        """
        Parameters
        ----------
        borrow : TODO
            TODO
        """
        return np.transpose(self._filters.get_value(borrow=borrow),
                            (0, 2, 3, 1))

    def lmul(self, x):
        """
        .. todo::

            WRITEME properly

        dot(x, A)

        This method overrides the original Conv2D lmul to make it work
        with arbitrary axis orders

        Parameters
        ----------
        x : TODO
            TODO
        """
        # x must be formatted as batch index, channel, topo dim 0, topo dim 1
        # for use with conv2d, so check what the current input space format is
        assert x.ndim == 4
        axes = self._input_space.axes
        assert len(axes) == 4

        op_axes = ('b', 'c', 0, 1)

        if tuple(axes) != op_axes:
            x = x.dimshuffle(*[axes.index(ax) for ax in op_axes])

        # The calling format has to be changed
        img = gpu_contiguous(x)
        kerns = gpu_contiguous(self._filters)
        shape = GpuDnnConv.get_out_shape(
            img.shape, kerns.shape, self._border_mode, self._subsample)
        rval = gpu_alloc_empty(*shape)
        desc = self._desc(img.shape, kerns.shape)
        rval = self._conv_op(img, kerns, rval, desc)

        # Format the output based on the output space
        axes = self._output_axes
        assert len(axes) == 4

        if tuple(self._output_axes) != op_axes:
            rval = rval.dimshuffle(*[op_axes.index(ax) for ax in
                                     self._output_axes])

        return rval

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        batch_size : TODO
            TODO
        """
        self._img_shape = tuple([batch_size] + list(self._img_shape[1:]))

def make_random_conv2D(irange, rng=None, *args, **kwargs):
    return OrigConv2D.make_random_conv2D(irange, rng, cls=Cudnn2D, *args, **kwargs)

def make_normal_conv2D(istd, rng=None, *args, **kwargs):
    return OrigConv2D.make_normal_conv2D(istd, rng, cls=Cudnn2D, *args, **kwargs)

def make_sparse_random_conv2D(num_nonzero, rng=None, *args, **kwargs):
    return OrigConv2D.make_sparse_random_conv2D(num_nonzero, rng, cls=Cudnn2D, *args, **kwargs)

