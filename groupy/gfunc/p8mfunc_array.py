
import groupy.garray.p8m_array as p8ma
from groupy.gfunc.gfuncarray import GFuncArray
import numpy as np


class P8MFuncArray(GFuncArray):

    def __init__(self, v, umin=None, umax=None, vmin=None, vmax=None):

        if umin is None or umax is None or vmin is None or vmax is None:
            if not (umin is None and umax is None and vmin is None and vmax is None):
                raise ValueError('Either all or none of umin, umax, vmin, vmax must equal None')

            # If (u, v) ranges are not given, determine them from the shape of v,
            # assuming the grid is centered.
            nu, nv = v.shape[-2:]

            hnu = nu // 2
            hnv = nv // 2

            umin = -hnu
            umax = hnu
            vmin = -hnv
            vmax = hnv

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2g = p8ma.meshgrid(
            m=p8ma.m_range(),
            r=p8ma.r_range(0, 8),
            u=p8ma.u_range(self.umin, self.umax + 1),
            v=p8ma.v_range(self.vmin, self.vmax + 1)
        )

        if v.shape[-3] == 16:
            i2g = i2g.reshape(16, i2g.shape[-2], i2g.shape[-1])
            self.flat_stabilizer = True
        else:
            self.flat_stabilizer = False

        super(P8MFuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gfloat = g.reparameterize('float').data.copy()
        gfloat[..., 2] -= self.umin
        gfloat[..., 3] -= self.vmin

        if self.flat_stabilizer:
            gfloat[..., 1] += gfloat[..., 0] * 8
            gfloat = gfloat[..., 1:]

        return gfloat
