
import numpy as np
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.Z2_array import Z2Array

# A transformation in p8m can be coded using four floats:
# m in {0, 1}, mirror reflection in the second translation axis or not
# r in {0, 1, 2, 3, 4, 5, 6, 7}, the rotation index
# u, translation along the first spatial axis
# v, translation along the second spatial axis
# We will always store these in the order (m, r, u, v).
# This is called the 'float' parameterization of p8m.

# A matrix representation of this group is given by
# T(u, v) M(m) R(r)
# where
# T = [[ 1, 0, u],
#      [ 0, 1, v],
#      [ 0, 0, 1]]
# M = [[ (-1) ** m, 0, 0],
#      [ 0,         1, 0],
#      [ 0,         0, 1]]
# R = [[ cos(r pi / 4), -sin(r pi /4), 0],
#      [ sin(r pi / 4), cos(r pi / 4), 0],
#      [ 0,             0,             1]]
# This is called the 'hmat' (homogeneous matrix) parameterization of p8m.

# The matrix representation is easier to work with when multiplying and inverting group elements,
# while the float parameterization is required when indexing gfunc on p8m.


class P8MArray(MatrixGArray):

    parameterizations = ['float', 'hmat']
    _g_shapes = {'float': (4,), 'hmat': (3, 3)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'p8m'

    def __init__(self, data, p='float'):
        data = np.asarray(data)
        assert data.dtype == np.float
        assert (p == 'float' and data.shape[-1] == 4) or (p == 'hmat' and data.shape[-2:] == (3, 3))

        self._left_actions[P8MArray] = self.__class__.left_action_hmat
        self._left_actions[Z2Array] = self.__class__.left_action_hvec

        self._reparameterizations[('float', 'hmat')] = self.float2hmat
        self._reparameterizations[('hmat', 'float')] = self.hmat2float

        super(P8MArray, self).__init__(data, p)
    
    def __eq__(self, other):
        if isinstance(other, self.__class__) or isinstance(self, other.__class__):
            return np.isclose(self.data, other.reparameterize(self.p).data).all(axis=-1)
        else:
            return NotImplemented
    
    def inv(self):
        mat_p = 'mat' if 'mat' in self.parameterizations else 'hmat'
        self_mat = self.reparameterize(mat_p).data
        self_mat_inv = np.linalg.inv(self_mat)
        return self.factory(data=self_mat_inv, p=mat_p).reparameterize(self.p)

    def float2hmat(self, float_data):
        m = float_data[..., 0]
        r = float_data[..., 1]
        u = float_data[..., 2]
        v = float_data[..., 3]
        out = np.zeros(float_data.shape[:-1] + (3, 3), dtype=np.float)
        out[..., 0, 0] = np.cos(0.25 * np.pi * r) * (-1) ** m
        out[..., 0, 1] = -np.sin(0.25 * np.pi * r) * (-1) ** m
        out[..., 0, 2] = u
        out[..., 1, 0] = np.sin(0.25 * np.pi * r)
        out[..., 1, 1] = np.cos(0.25 * np.pi * r)
        out[..., 1, 2] = v
        out[..., 2, 2] = 1.
        return out

    def hmat2float(self, hmat_data):
        neg_det_r = hmat_data[..., 1, 0] * hmat_data[..., 0, 1] - hmat_data[..., 0, 0] * hmat_data[..., 1, 1]
        s = hmat_data[..., 1, 0]
        c = hmat_data[..., 1, 1]
        u = hmat_data[..., 0, 2]
        v = hmat_data[..., 1, 2]
        m = np.round(neg_det_r + 1) // 2
        r = np.round(np.arctan2(s, c) / np.pi * 4) % 8

        out = np.zeros(hmat_data.shape[:-2] + (4,), dtype=np.float)
        out[..., 0] = m
        out[..., 1] = r
        out[..., 2] = u
        out[..., 3] = v
        out[np.isclose(out, 0)] = 0
        return out


def identity(shape=(), p='float'):
    e = P8MArray(np.zeros(shape + (4,), dtype=np.float), 'float')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size + (4,), dtype=np.float)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 8, size)
    data[..., 2] = np.random.randint(minu, maxu, size)
    data[..., 3] = np.random.randint(minv, maxv, size)
    return P8MArray(data=data, p='float')


def rotation(r, center=(0, 0)):
    r = np.asarray(r)
    center = np.asarray(center)

    rdata = np.zeros(r.shape + (4,), dtype=np.float)
    rdata[..., 1] = r
    r0 = P8MArray(rdata)

    tdata = np.zeros(center.shape[:-1] + (4,), dtype=np.float)
    tdata[..., 2:] = center
    t = P8MArray(tdata)

    return t * r0 * t.inv()


def mirror_u(shape=None):
    shape = shape if shape is not None else ()
    mdata = np.zeros(shape + (4,), dtype=np.float)
    mdata[0] = 1
    return P8MArray(mdata)


def mirror_v(shape=None):
    hm = mirror_u(shape)
    r = rotation(2)
    return r * hm * r.inv()


def m_range(start=0, stop=2):
    assert stop > 0
    assert stop <= 2
    assert start >= 0
    assert start < 2
    assert start < stop
    m = np.zeros((stop - start, 4), dtype=np.float)
    m[:, 0] = np.arange(start, stop)
    return P8MArray(m)


def r_range(start=0, stop=8, step=1):
    assert stop > 0
    assert stop <= 8
    assert start >= 0
    assert start < 8
    assert start < stop
    m = np.zeros((stop - start, 4), dtype=np.float)
    m[:, 1] = np.arange(start, stop, step)
    return P8MArray(m)


def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.float)
    m[:, 2] = np.arange(start, stop, step)
    return P8MArray(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.float)
    m[:, 3] = np.arange(start, stop, step)
    return P8MArray(m)


def meshgrid(m=m_range(), r=r_range(), u=u_range(), v=v_range()):
    m = P8MArray(m.data[:, None, None, None, ...], p=m.p)
    r = P8MArray(r.data[None, :, None, None, ...], p=r.p)
    u = P8MArray(u.data[None, None, :, None, ...], p=u.p)
    v = P8MArray(v.data[None, None, None, :, ...], p=v.p)
    return u * v * m * r


# def gmeshgrid(*args):
#    out = identity()
#    for i in range(len(args)):
#        slices = [None if j != i else slice(None) for j in range(len(args))] + [Ellipsis]
#        d = args[i].data[slices]
#        print i, slices, d.shape
#        out *= P4MArray(d, p=args[i].p)
#
#    return out
