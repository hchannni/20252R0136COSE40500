import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import math

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : Pytorch tensor
        N-d Pytorch tensor.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.
    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import torch
    >>> A = torch.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = torch.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = torch.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not torch.is_tensor(arr_in):
        raise TypeError("`arr_in` must be a pytorch tensor")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = torch.tensor(arr_in.shape)
    window_shape = torch.tensor(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    # window_strides = torch.tensor(arr_in.stride())
    window_strides = arr_in.stride()

    indexing_strides = arr_in[slices].stride()

    win_indices_shape = torch.div(arr_shape - window_shape
                          , torch.tensor(step), rounding_mode = 'floor') + 1
    
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = torch.as_strided(arr_in, size=new_shape, stride=strides)
    return arr_out

class nn_convolutional_layer:

    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):
        
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                                  size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.input_size = input_size
        self.cache = None

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()

    def _parse_stride_padding(self, stride, padding):
        if isinstance(stride, numbers.Number):
            stride = (int(stride), int(stride))
        if isinstance(padding, numbers.Number):
            padding = (int(padding), int(padding))
        if len(stride) != 2 or len(padding) != 2:
            raise ValueError("`stride` and `padding` must be int or tuple of length 2.")
        return stride, padding

    def forward(self, x, stride=1, padding=0):
        stride, padding = self._parse_stride_padding(stride, padding)
        N, C, H, W = x.shape
        KH, KW = self.W.shape[2], self.W.shape[3]

        x_padded = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
        x_unfold = F.unfold(x_padded, kernel_size=(KH, KW), stride=stride)  # (N, C*KH*KW, L)
        w_flat = self.W.reshape(self.W.shape[0], -1)  # (out_ch, C*KH*KW)

        out = torch.bmm(w_flat.unsqueeze(0).expand(N, -1, -1), x_unfold)  # (N, out_ch, L)
        out = out + self.b.reshape(1, self.W.shape[0], 1)

        H_out = (H + 2 * padding[0] - KH) // stride[0] + 1
        W_out = (W + 2 * padding[1] - KW) // stride[1] + 1
        out = out.view(N, self.W.shape[0], H_out, W_out)

        self.cache = {
            "x_shape": x.shape,
            "x_unfold": x_unfold,
            "stride": stride,
            "padding": padding,
            "x_padded_shape": x_padded.shape,
        }

        return out

    def backward(self, d_out):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward.")

        x_shape = self.cache["x_shape"]
        x_unfold = self.cache["x_unfold"]
        stride = self.cache["stride"]
        padding = self.cache["padding"]
        x_padded_shape = self.cache["x_padded_shape"]

        N, _, H_out, W_out = d_out.shape
        out_ch = self.W.shape[0]
        w_flat = self.W.reshape(out_ch, -1)

        d_out_flat = d_out.reshape(N, out_ch, -1)  # (N, out_ch, L)

        db = d_out_flat.sum(dim=(0, 2)).reshape(1, out_ch, 1, 1)

        dW_batch = torch.bmm(d_out_flat, x_unfold.transpose(1, 2))  # (N, out_ch, C*KH*KW)
        dW = dW_batch.sum(dim=0).reshape_as(self.W)

        dx_unfold = torch.bmm(w_flat.t().unsqueeze(0).expand(N, -1, -1), d_out_flat)  # (N, C*KH*KW, L)

        H_padded, W_padded = x_padded_shape[2], x_padded_shape[3]
        dx_padded = F.fold(
            dx_unfold,
            output_size=(H_padded, W_padded),
            kernel_size=(self.W.shape[2], self.W.shape[3]),
            stride=stride,
        )

        pad_h, pad_w = padding
        dx = dx_padded[:, :, pad_h : pad_h + x_shape[2], pad_w : pad_w + x_shape[3]]

        return dx, dW, db

    

class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        self.stride = stride
        self.pool_size = pool_size
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        p = self.pool_size
        s = self.stride

        x_unfold = F.unfold(x, kernel_size=p, stride=s)  # (N, C*p*p, L)
        x_unfold = x_unfold.view(N, C, p * p, -1)  # (N, C, p^2, L)
        out, idx = torch.max(x_unfold, dim=2)  # (N, C, L)

        H_out = (H - p) // s + 1
        W_out = (W - p) // s + 1
        out = out.view(N, C, H_out, W_out)

        self.cache = {
            "x_shape": x.shape,
            "idx": idx,
            "p": p,
            "s": s,
            "L": x_unfold.shape[-1],
        }

        return out

    def backward(self, d_out):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward.")

        x_shape = self.cache["x_shape"]
        idx = self.cache["idx"]
        p = self.cache["p"]
        s = self.cache["s"]
        L = self.cache["L"]

        N, C, H, W = x_shape
        d_out_flat = d_out.view(N, C, -1)  # (N, C, L)

        grad_unfold = torch.zeros((N, C, p * p, L), device=d_out.device, dtype=d_out.dtype)
        grad_unfold.scatter_(2, idx.unsqueeze(2), d_out_flat.unsqueeze(2))

        grad_unfold = grad_unfold.view(N, C * p * p, L)
        dx = F.fold(grad_unfold, output_size=(H, W), kernel_size=p, stride=s)

        return dx

    
def gradient_check_conv():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 5, dtype=torch.double)

    layer = nn_convolutional_layer(f_height=3, f_width=3, input_size=5, in_ch_size=2, out_ch_size=3)
    layer.W = layer.W.double()
    layer.b = layer.b.double()

    stride = (2, 1)
    padding = (1, 1)

    out_custom = layer.forward(x, stride=stride, padding=padding)
    dout = torch.randn_like(out_custom)
    dx_custom, dW_custom, db_custom = layer.backward(dout)

    x_ref = x.clone().detach().requires_grad_(True)
    W_ref = layer.W.clone().detach().requires_grad_(True)
    b_ref = layer.b.clone().detach().squeeze().requires_grad_(True)

    out_ref = F.conv2d(x_ref, W_ref, bias=b_ref, stride=stride, padding=padding)
    loss = (out_ref * dout).sum()
    loss.backward()

    dx_ref = x_ref.grad
    dW_ref = W_ref.grad
    db_ref = b_ref.grad.reshape(1, -1, 1, 1)

    def rel_err(a, b):
        return (a - b).abs().max() / (b.abs().max().clamp_min(1e-9))

    print("conv grad check:")
    print("  dx rel error :", rel_err(dx_custom, dx_ref).item())
    print("  dW rel error :", rel_err(dW_custom, dW_ref).item())
    print("  db rel error :", rel_err(db_custom, db_ref).item())


def gradient_check_pool():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 5, dtype=torch.double, requires_grad=True)
    pool = nn_max_pooling_layer(pool_size=2, stride=2)

    out_custom = pool.forward(x)
    dout = torch.randn_like(out_custom)
    dx_custom = pool.backward(dout)

    out_ref = F.max_pool2d(x, kernel_size=2, stride=2)
    loss = (out_ref * dout).sum()
    loss.backward()

    dx_ref = x.grad

    rel_error = (dx_custom - dx_ref).abs().max() / dx_ref.abs().max().clamp_min(1e-9)
    print("pool grad check:")
    print("  dx rel error :", rel_error.item())


if __name__ == "__main__":

    batch_size = 8
    input_size = 32
    filter_width = 3
    filter_height = filter_width
    in_ch_size = 3
    num_filters = 8

    std = 1e0
    dt = 1e-3

    num_test = 50

    err_fwd = 0
    err_pool = 0


    print('conv test')
    for i in range(num_test):
        cnv = nn_convolutional_layer(filter_height, filter_width, input_size,
                                   in_ch_size, num_filters)

        test_conv_layer = nn.Conv2d(in_channels=in_ch_size, out_channels=num_filters,
                                kernel_size = (filter_height, filter_width))
        
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
        
        with torch.no_grad():
            
            out = cnv.forward(x)
            W,b = cnv.get_weights()
            test_conv_layer.weight = nn.Parameter(W)
            test_conv_layer.bias = nn.Parameter(torch.squeeze(b))
            test_out = test_conv_layer(x)
            
            err=torch.norm(test_out - out)/torch.norm(test_out)
            err_fwd+= err
    
    stride = 2
    pool_size = 2
    
    print('pooling test')
    for i in range(num_test):
        mpl = nn_max_pooling_layer(pool_size=pool_size, stride=stride)
        
        test_pooling_layer = nn.MaxPool2d(kernel_size=(pool_size,pool_size), stride=stride)
        
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
        
        with torch.no_grad():
            out = mpl.forward(x)
            test_out = test_pooling_layer(x)
            
            err=torch.norm(test_out - out)/torch.norm(test_out)
            err_pool+= err

    print('accuracy results')
    print('forward accuracy', 100 - err_fwd/num_test*100, '%')
    print('pooling accuracy', 100 - err_pool/num_test*100, '%')

    gradient_check_conv()
    gradient_check_pool()