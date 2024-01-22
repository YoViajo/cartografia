#
# Copyright (c) 2019 Leland Brown.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
# HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
#

import numpy as np
import scipy.fftpack
import scipy.fftpack._fftpack

import weakref


try:
    range = xrange  # for Python 2; works only if values are within range of C int
except NameError:
    pass


def transpose_inplace(a, callback=lambda n, d: None, pct1=0, dpct=100):
    # placeholder for actual inplace algorithm
    newa = np.copy(a.transpose((1, 0)))
    a.shape = newa.shape
    #a[...] = newa
    ncols = newa.shape[0]
    numer = ncols * pct1
    denom = ncols * 100
    for i in range(ncols):
        a[i,...] = newa[i,...]
        callback(numer + i * dpct, denom)


class Dct(object):
    __slots__ = (
        "_n", "_m", "_type", "_dtype",      # defining parameters
        "_w1", "_w2", "_wb", "_wc", "_ws",  # precomputed arrays
        "_xh", "_t1", "_t2", "_wa",         # temporary storage arrays
        "__weakref__"
    )

    def __init__(self, n, type=2, dtype=np.float64):
        if type != 2:
            raise NotImplementedError(
                "Only type 2 DCT currently supported (use idct for type 3)")

        self._n = n
        self._type  = type
        self._dtype = dtype

        cost, m = self.bluestein_cost(n)

        if self.fft_cost(n) < cost:
            # Bluestein's algorithm may be slower - don't use it for this n
            self._m = 0
            return

        self._m = m

        n2 = n + n

        dt = np.pi / n2

        theta = dt * np.arange(n)
        w = np.cos(theta).astype(dtype)
        # w[i]  = cos(pi/2*i/n)
        # w[-i] = sin(pi/2*i/n)

        w1 = np.empty(n, dtype)
        w2 = np.empty(n, dtype)

        w1[0] = w2[0] = 1.0
        w1[1:] = w[:0:-1] + w[1:]
        w2[1:] = w[:0:-1] - w[1:]

        w = None

        wcs = np.empty((2, n), dtype)
        wc = wcs[0]
        ws = wcs[1]

        theta = (2.0 * dt) * (np.arange(n)**2 % n2)
        wc[:] = np.cos(theta)
        ws[:] = np.sin(theta)

        theta = None

        wb = np.empty((2, m), dtype)

        wb[:,0:n]   = wcs
        wb[:,n:1-n] = 0.0
        wb[:,1-n:]  = wcs[:,:0:-1]

        wb = scipy.fftpack.rfft(x=wb, overwrite_x=True)

        wb1 = wb[0]
        wb2 = wb[1]

        t            = wb1[1:-1:2] + wb2[2::2]
        wb1[1:-1:2] -= wb2[2::2]
        wb2[2::2]    = wb2[1:-1:2] - wb1[2::2]
        wb2[1:-1:2] += wb1[2::2]
        wb1[2::2]    = t

        t = None

        self._w1 = w1
        self._w2 = w2
        self._wb = wb
        self._wc = wc
        self._ws = ws

        # allocate temporary arrays

        self._xh = np.empty(n, dtype)
        self._t1 = np.empty(m, dtype)
        self._t2 = np.empty(m, dtype)
        self._wa = np.empty((2, m), dtype)

        weakref.finalize(self, lambda: None)  # placeholder for cleanup behavior at garbage collection

    @staticmethod
    def _clear_cache(m, type, dtype):
        # clean up scipy.fftpack cache memory; may not be thread-safe
        if dtype == np.float64:
            if m:
                scipy.fftpack._fftpack.destroy_drfft_cache()
            elif type == 1:
                scipy.fftpack._fftpack.destroy_ddct1_cache()
            else:
                scipy.fftpack._fftpack.destroy_ddct2_cache()
        else:
            if m:
                scipy.fftpack._fftpack.destroy_rfft_cache()
            elif type == 1:
                scipy.fftpack._fftpack.destroy_dct1_cache()
            else:
                scipy.fftpack._fftpack.destroy_dct2_cache()

    def close(self, clear_cache=True):
        if clear_cache:
            self._clear_cache(self._m, self._type, self._dtype)

        # mark object as closed, disable further use
        self._n = 0

        # clean up workspace arrays
        self._w1 = self._w2 = self._wb = self._wc = self._ws = None
        self._xh = self._t1 = self._t2 = self._wa = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(clear_cache=True)

    def bluestein(self):
        return self._m != 0

    @staticmethod
    def fft_cost(n):
        # find factors of n and estimate time cost for FFTPACK

        cost = 0

        nl = n

        while nl & 3 == 0:
            nl >>= 2
            cost += 5  # factor of 4

        if nl & 1 == 0:
            nl >>= 1
            cost += 5  # factor of 2

        if nl == 1:
            return cost * n

        for ntry in range(3, nl + 1, 2):

            nq = nl // ntry
            if nq < ntry:
                cost += 3 + nl  # factor of nl
                break
            nr = nl - ntry * nq

            while nr == 0:
                cost += 3 + ntry  # factor of ntry
                if nq == 1:
                    return cost * n
                nl = nq
                nq = nl // ntry
                nr = nl - ntry * nq

        return cost * n

    @staticmethod
    def bluestein_cost(n):
        # If the convolution in _bluestein() is replaced with a faster algorithm,
        # then reduce the value of BLUESTEIN_THRESHOLD proportionally:
        BLUESTEIN_THRESHOLD = 20

        n2 = n + n
        m = 8
        mf = 5
        while m < n:
            m += m
            mf += 1
        m3 = (m * 3) >> 1  # try replacing one factor of 2 with 3
        if m3 >= n2:
            m = m3
        else:
            m += m
        mf >>= 1  # number of factors of 4, 2, and 3

        return BLUESTEIN_THRESHOLD * mf * m, m

    def _bluestein(self, xr, xi):
        n  = self._n
        wa = self._wa
        wb = self._wb
        wc = self._wc
        ws = self._ws
        xh = self._xh
        t1 = self._t1
        t2 = self._t2

        add = np.add
        sub = np.subtract
        mul = np.multiply

        wa1 = wa[0]
        wa2 = wa[1]

        u1 = wa1[:n]
        u2 = wa2[:n]

        # u1[:] = xr * wc + xi * ws

        mul(xr, wc, out=u1)
        mul(xi, ws, out=xh)
        add(u1, xh, out=u1)

        wa1[n:] = 0.0

        # u2[:] = xi * wc - xr * ws

        mul(xi, wc, out=u2)
        mul(xr, ws, out=xh)
        sub(u2, xh, out=u2)

        wa2[n:] = 0.0

        scipy.fftpack.rfft(x=wa, overwrite_x=True)  # assumes in-place transform

        wb1 = wb[0]
        wb2 = wb[1]

        # t1 = wa1 * wb2
        # t2 = wa2 * wb2
        # wa1[:] = wa1 * wb1 - t2
        # wa2[:] = wa2 * wb1 + t1

        mul(wa1, wb2, out=t1)
        mul(wa1, wb1, out=wa1)
        mul(wa2, wb2, out=t2)
        mul(wa2, wb1, out=wa2)
        sub(wa1, t2, out=wa1)
        add(wa2, t1, out=wa2)

        scipy.fftpack.irfft(x=wa, overwrite_x=True)  # assumes in-place transform

        # xr[:] = u1 * wc + u2 * ws

        mul(u1, wc, out=xr)
        mul(u2, ws, out=xh)
        add(xr, xh, out=xr)

        # xi[:] = u2 * wc - u1 * ws

        mul(u2, wc, out=xi)
        mul(u1, ws, out=xh)
        sub(xi, xh, out=xi)

    def cosqb2_bluestein(self, x):
        assert x.dtype == self._dtype

        n = self._n

        assert x.shape == (2, n)

        if not self._m:
            x = scipy.fftpack.dct(x=x, type=self._type, overwrite_x=True)
            x += x
            return

        add = np.add
        sub = np.subtract
        mul = np.multiply

        x1 = x[0]
        x2 = x[1]

        xh = self._xh
        w1 = self._w1
        w2 = self._w2

        x1a = x1[1:-1:2]
        x1b = x1[2::2]
        x2a = x2[1:-1:2]
        x2b = x2[2::2]
        xha = xh[1:-1:2]
        xhb = xh[2::2]

        # xh[2::2]   = x1[2::2] - x1[1:-1:2]
        # xh[1:-1:2] = x2[2::2] + x2[1:-1:2]

        sub(x1b, x1a, out=xhb)
        add(x2b, x2a, out=xha)

        # x2[2::2]  -= x2[1:-1:2]
        # x2[1:-1:2] = x1[2::2] + x1[1:-1:2]

        sub(x2b, x2a, out=x2b)
        add(x1b, x1a, out=x2a)

        x1[0] *= 2.0
        x2[0] *= 2.0

        ns2 = (n + 1) >> 1

        even = (n & 1 == 0)

        if even:
            x1[ns2] = x1[-1] * 2.0

        # x1[1:ns2]    = x2[1:-1:2] - x2[2::2]
        # x1[:-ns2:-1] = x2[1:-1:2] + x2[2::2]

        x1c = x1[1:ns2]
        x1d = x1[:-ns2:-1]
        x2c = x2[1:ns2]
        x2d = x2[:-ns2:-1]

        sub(x2a, x2b, out=x1c)
        add(x2a, x2b, out=x1d)

        if even:
            x2[ns2] = x2[-1] * 2.0

        # x2[1:ns2]    = xh[1:-1:2] + xh[2::2]
        # x2[:-ns2:-1] = xh[1:-1:2] - xh[2::2]

        add(xha, xhb, out=x2c)
        sub(xha, xhb, out=x2d)

        self._bluestein(x2, x1)

        # x1[1:] = x1[1:] * w1[1:] + x1[:0:-1] * w2[:0:-1]

        mul(x1, w2, out=xh)
        mul(x1, w1, out=x1)  # leaves x1[0] unchanged
        x1[1:] += xh[:0:-1]

        x1[0] *= 2.0

        # x2[1:] = x2[1:] * w1[1:] + x2[:0:-1] * w2[:0:-1]

        mul(x2, w2, out=xh)
        mul(x2, w1, out=x2)  # leaves x2[0] unchanged
        x2[1:] += xh[:0:-1]

        x2[0] *= 2.0

    def cosqf2_bluestein(self, x):
        assert x.dtype == self._dtype

        n = self._n

        assert x.shape == (2, n)

        if not self._m:
            x = scipy.fftpack.idct(x=x, type=self._type, overwrite_x=True)
            x += x
            return

        add = np.add
        sub = np.subtract
        mul = np.multiply

        x1 = x[0]
        x2 = x[1]

        xh = self._xh
        w1 = self._w1
        w2 = self._w2

        # x1[1:] = x1[1:] * w1[1:] - x1[:0:-1] * w2[:0:-1]

        mul(x1, w2, out=xh)
        mul(x1, w1, out=x1)  # leaves x1[0] unchanged
        x1[1:] -= xh[:0:-1]

        # x2[1:] = x2[1:] * w1[1:] - x2[:0:-1] * w2[:0:-1]

        mul(x2, w2, out=xh)
        mul(x2, w1, out=x2)  # leaves x2[0] unchanged
        x2[1:] -= xh[:0:-1]

        self._bluestein(x1, x2)

        x1a = x1[1:-1:2]
        x1b = x1[2::2]
        x2a = x2[1:-1:2]
        x2b = x2[2::2]
        xha = xh[1:-1:2]
        xhb = xh[2::2]

        ns2 = (n + 1) >> 1

        x1c = x1[1:ns2]
        x1d = x1[:-ns2:-1]
        x2c = x2[1:ns2]
        x2d = x2[:-ns2:-1]

        # xh[1:-1:2] = x1[1:ns2] + x1[:-ns2:-1]
        # xh[2::2]   = x1[1:ns2] - x1[:-ns2:-1]

        add(x1c, x1d, out=xha)
        sub(x1c, x1d, out=xhb)

        even = (n & 1 == 0)

        if even:
            x1[-1] = x1[ns2] * 2.0

        # x1[1:-1:2] = x2[1:ns2] + x2[:-ns2:-1]
        # x1[2::2]   = x2[1:ns2] - x2[:-ns2:-1]

        add(x2c, x2d, out=x1a)
        sub(x2c, x2d, out=x1b)

        if even:
            x2[-1] = x2[ns2] * 2.0

        # x2[1:-1:2]  = x1[1:-1:2] + xh[2::2]
        # x2[2::2]    = x1[1:-1:2] - xh[2::2]

        add(x1a, xhb, out=x2a)
        sub(x1a, xhb, out=x2b)

        # x1[1:-1:2]  = xh[1:-1:2] - x1[2::2]
        # x1[2::2]   += xh[1:-1:2]

        sub(xha, x1b, out=x1a)
        add(xha, x1b, out=x1b)

        x1[0] *= 2.0
        x2[0] *= 2.0


def dct2(x, type=2, transpose=False, callback=lambda n,d: None):
    assert x.dtype.char in np.typecodes["Float"]

    nrows, ncols = x.shape

    row_major = x.strides[0] >= x.strides[1]

    if not row_major:
        x = x.transpose()

    callback(0, 100)

    dct = Dct(ncols, type, np.float64)
    a = np.empty((2, ncols), np.float64)
    odd = nrows & 1

    denom = nrows * 100

    if odd:
        a[0] = x[0]
        a[1] = a[0]
        dct.cosqb2_bluestein(a)
        x[0] = a[1]
        callback(40, denom)

    for i in range(odd,nrows,2):
        i2 = i + 2
        a[:] = x[i:i2]
        dct.cosqb2_bluestein(a)
        x[i:i2] = a
        callback(i2 * 40, denom)

    dct.close()

    transpose_inplace(x, callback, 40, 20)

    callback(60, 100)

    dct = Dct(nrows, type, np.float64)
    a = np.empty((2, nrows), np.float64)
    odd = ncols & 1

    numer = ncols * 60
    denom = ncols * 100

    if odd:
        a[0] = x[0]
        a[1] = a[0]
        dct.cosqb2_bluestein(a)
        x[0] = a[1]
        callback(numer + 40, denom)

    for i in range(odd,ncols,2):
        i2 = i + 2
        a[:] = x[i:i2]
        dct.cosqb2_bluestein(a)
        x[i:i2] = a
        callback(numer + i2 * 40, denom)

    dct.close()

    callback(100, 100)

    a = None
    dct = None

    # x *= 0.25  # make scaling consistent with scipy.fftpack

    if row_major ^ bool(transpose):
        x = x.transpose()


def idct2(x, type=2, transpose=False, callback=lambda n,d: None):
    assert x.dtype.char in np.typecodes["Float"]

    nrows, ncols = x.shape

    row_major = x.strides[0] >= x.strides[1]

    if not row_major:
        x = x.transpose()

    callback(0, 100)

    dct = Dct(ncols, type, np.float64)
    a = np.empty((2, ncols), np.float64)
    odd = nrows & 1

    denom = nrows * 100

    if odd:
        a[0] = x[0]
        a[1] = a[0]
        dct.cosqf2_bluestein(a)
        x[0] = a[1]
        callback(40, denom)

    for i in range(odd, nrows, 2):
        i2 = i + 2
        a[:] = x[i:i2]
        dct.cosqf2_bluestein(a)
        x[i:i2] = a
        callback(i2 * 40, denom)

    dct.close()

    transpose_inplace(x, callback, 40, 20)

    callback(60, 100)

    dct = Dct(nrows, type, np.float64)
    a = np.empty((2, nrows), np.float64)
    odd = ncols & 1

    numer = ncols * 60
    denom = ncols * 100

    if odd:
        a[0] = x[0]
        a[1] = a[0]
        dct.cosqf2_bluestein(a)
        x[0] = a[1]
        callback(numer + 40, denom)

    for i in range(odd, ncols, 2):
        i2 = i + 2
        a[:] = x[i:i2]
        dct.cosqf2_bluestein(a)
        x[i:i2] = a
        callback(numer + i2 * 40, denom)

    dct.close()

    callback(100, 100)

    a = None
    dct = None

    # x *= 0.25  # make scaling consistent with scipy.fftpack

    if row_major ^ bool(transpose):
        x = x.transpose()
