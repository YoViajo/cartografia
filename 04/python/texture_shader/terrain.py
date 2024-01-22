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

from try_imports import gdal, osr

from preferences import PREVIEW_DENSITY

from constants import NUM_DISTORTION_SAMPLES
from constants import INTERP_AREA_AS_LOG
from constants import USE_CUTOFF_FREQUENCY
from constants import TERRAIN_OPERATOR_METHOD

from georeference import GeoReference

from dct import dct2, idct2

import scipy.interpolate
import numpy as np

import math
#import sys


try:
    range = xrange  # for Python 2; works only if values are within range of C int
except NameError:
    pass


class DistortionInfo():
    def __init__(self, geo_ref):
        nrows = geo_ref.nrows
        ncols = geo_ref.ncols

        index = np.arange(NUM_DISTORTION_SAMPLES)
        self.sample_rows = (
            index * (nrows - 1) / (NUM_DISTORTION_SAMPLES - 1.0)
        )
        self.sample_cols = (
            index * (ncols - 1) / (NUM_DISTORTION_SAMPLES - 1.0)
        )
        cols, rows = np.meshgrid(self.sample_cols, self.sample_rows)

        # stack rows, cols along 3rd dimension
        rowcol = np.dstack((rows, cols))
        rows = cols = None  # release memory

        # flatten 1st & 2nd dimensions into 1st
        rowcol = rowcol.reshape(-1, 1, 2)

        # expand along 2nd dimension to give 4 corners for each pixel
        corners = np.array([[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
        rowcol = rowcol + corners

        # convert to spatial coordinates (ECEF)
        rowcol = rowcol.reshape(-1, 2)
        xyz = geo_ref.pixel_to_ecef_array(rowcol)

        rowcol = None  # release memory

        # reshape to xyz[row_sample, col_sample, corner, coordinate (x/y/z)]
        xyz = xyz.reshape(NUM_DISTORTION_SAMPLES, NUM_DISTORTION_SAMPLES, 4, 3)

        # diagonal vectors of each sample pixel in ECEF
        diag1 = xyz[:, :, 1, :] - xyz[:, :, 0, :]
        diag2 = xyz[:, :, 3, :] - xyz[:, :, 2, :]

        xyz = None  # release memory

        # vectors down and across each sample pixel in ECEF
        drow = (diag1 - diag2) * 0.5
        dcol = (diag1 + diag2) * 0.5

        diag1 = diag2 = None  # release memory

        # find area of each sample pixel (to be interpolated later to all pixels)
        normal = np.cross(drow, dcol)
        self.pixel_area_sqr = np.sum(normal * normal, -1)

        normal = None  # release memory

        # distortion matrix components for each sample pixel
        a = np.sum(dcol * dcol, -1)
        b = np.sum(drow * dcol, -1)
        c = np.sum(drow * drow, -1)

        drow = dcol = None  # release memory

        # find largest eigenvalue of each matrix
        eigenvalue = 0.5 * ((a + c) + np.sqrt((c - a) ** 2 + 4.0 * b ** 2))
        eigenvalue[eigenvalue == 0.0] = 1.0  # avoid NaNs if drow = dcol = zero

        # normalize each matrix so largest eigenvalue is 1
        a /= eigenvalue
        b /= eigenvalue
        c /= eigenvalue

        # find average matrix, normalized so determinant is 1
        self.xxres = np.sum(a)
        self.xyres = np.sum(b)
        self.yyres = np.sum(c)
        det = self.xxres * self.yyres - self.xyres * self.xyres
        if det <= 0.0:
            # this should never happen, but provide a default just in case
            self.xxres = self.yyres = 1.0
            self.xyres = 0.0
        else:
            sqrt_det = math.sqrt(det)
            self.xxres /= sqrt_det
            self.xyres /= sqrt_det
            self.yyres /= sqrt_det

        # TODO: check max distortion of a,b,c against xxres, xyres, yyres


class TextureOperator:
    def __init__(self, detail, ncols, nrows, xxscale, yyscale, xyscale):
        # Note: data array is in transposed layout (ncols x nrows)

        # __scale = actual __ scale * pixel area, etc.

        m2 = ncols + ncols
        n2 = nrows + nrows
        self._factor = 0.25 / (n2 * m2)  # DCT normalization factor

        # multiplying factor for fractional Laplacian
        self._factor *= math.pow(2.0 * math.pi, detail)

        self._power = detail * 0.5

        xstep = 1.0 / m2
        ystep = 1.0 / n2

        if USE_CUTOFF_FREQUENCY:
            minxfreq2 = (xstep ** 2) * xxscale
            minyfreq2 = (ystep ** 2) * yyscale

            self._adjust = 4.0 * max(minxfreq2, minyfreq2)
        else:
            self._adjust = 0.0  # for scale-invariant fractional Laplacian

        # Use approximation to bicubic spline interpolator

        nux = xstep * np.arange(m2 + 1)  # 0 to 1 cycles/pixel
        cosx = np.cos(math.pi * nux)
        tempx = cosx * 0.5 + 0.5
        self._splinex = (
            (tempx * tempx) * ((cosx + 2.0) / (cosx * cosx * 2.0 + 1.0))
        )

        nuy = ystep * np.arange(n2 + 1)  # 0 to 1 cycles/pixel
        cosy = np.cos(math.pi * nuy)
        tempy = cosy * 0.5 + 0.5
        self._spliney = (
            (tempy * tempy) * ((cosy + 2.0) / (cosy * cosy * 2.0 + 1.0))
        )

        self._x = nux
        self._y = nuy

        self._xx = (nux ** 2) * xxscale
        self._yy = (nuy ** 2) * yyscale

        # only apply factor to one of _x, _y
        # since they will be multiplied together
        self._x *= 2.0 * xyscale

        if TERRAIN_OPERATOR_METHOD == 0:
            # model Laplacian using cubic spline surface

            self._separablexx = (
                self._splinex[:ncols]       * self._xx[:ncols] +
                self._splinex[:-1-ncols:-1] * self._xx[:-1-ncols:-1]
            )
            self._separableyy = (
                self._spliney[:nrows]       * self._yy[:nrows] +
                self._spliney[:-1-nrows:-1] * self._yy[:-1-nrows:-1]
            )

            self._separablex = (
                self._splinex[:ncols]       * self._x[:ncols] +
                self._splinex[:-1-ncols:-1] * self._x[:-1-ncols:-1]
            )
            self._separabley = (
                self._spliney[:nrows]       * self._y[:nrows] +
                self._spliney[:-1-nrows:-1] * self._y[:-1-nrows:-1]
            )
        else:
            self.log1 = np.empty_like(self._spliney)
            self.log2 = np.empty_like(self._spliney)
            self.log3 = np.empty_like(self._spliney)
            self.log4 = np.empty_like(self._spliney)

    def __call__(self, data, col, nrows):
        data = data[col]

        if TERRAIN_OPERATOR_METHOD == 0:
            # model Laplacian using cubic spline surface

            data *= np.power(
                self._adjust +
                self._separablexx[col] + self._separableyy[:nrows] +
                self._separablex[col]  * self._separabley[:nrows],
                self._power
            )

        else:
            # model infinitesimal fractional Laplacian using cubic spline

            log1 = self.log1
            log2 = self.log2
            log3 = self.log3
            log4 = self.log4

            log1[:] = self._splinex[col]    * self._spliney[:nrows]
            log2[:] = self._splinex[col]    * self._spliney[-1:-1-nrows:-1]
            log3[:] = self._splinex[-1-col] * self._spliney[:nrows]
            log4[:] = self._splinex[-1-col] * self._spliney[-1:-1-nrows:-1]

            log1 *= np.log(
                self._adjust +
                self._xx[col] + self._yy[:nrows] +
                self._x[col]  * self._y[:nrows]
            )
            log2 *= np.log(
                self._adjust +
                self._xx[col] + self._yy[-1:-1-nrows:-1] +
                self._x[col]  * self._y[-1:-1-nrows:-1]
            )
            log3 *= np.log(
                self._adjust +
                self._xx[-1-col] + self._yy[:nrows] +
                self._x[-1-col]  * self._y[:nrows]
            )
            log4 *= np.log(
                self._adjust +
                self._xx[-1-col] + self._yy[-1:-1-nrows:-1] +
                self._x[-1-col]  * self._y[-1:-1-nrows:-1]
            )

            data *= np.exp(((log1 + log4) + (log2 + log3)) * self._power)

        data *= self._factor

        if col == 0:
            data[0] = 0.0  # set "DC" component to zero


class SubProgress():
    def __init__(self, callback, pct1, pct2):
        self.callback = callback
        self.pct1 = pct1
        self.dpct = pct2 - pct1

    def __call__(self, numer, denom):
        self.callback(denom * self.pct1 + numer * self.dpct, denom * 100)


class TerrainRaster():
    def __init__(self, elevation, geo_ref, min_elev, max_elev, has_voids):
        # Note: geo_ref can have more pixels than elevation array but should
        # cover the same geographic extent (measured from outside pixel edges)
        self.nrows, self.ncols = elevation.shape
        self.elevation = elevation
        self.geo_ref = geo_ref
        self.distortion = DistortionInfo(geo_ref)
        self.min_elev = min_elev
        self.max_elev = max_elev
        self.has_voids = has_voids
        self.texture = None
        self._fourier = None
        self._prev_detail = None

    def grayscale(self, contrast, brightness):
        if self.texture is None:
            # must call texture_shade() first
            return None
        
        grayscale = self.texture.copy()

        # make contrast and brightness adjustments
        grayscale *= math.pow(2.0, contrast / 8.0)
        grayscale += brightness / 20.0

        # convert to range (-1, +1)
        # grayscale = grayscale / sqrt(grayscale ** 2 + 1)
        # alternative conversion: np.tanh(preview_gray, preview_gray)
        temp = np.empty_like(grayscale[0])
        for i in range(self.nrows):
            # process by rows to reduce memory usage
            temp[:] = grayscale[i]
            np.square(temp, temp)
            temp += 1.0
            np.sqrt(temp, temp)
            grayscale[i] /= temp

        # convert to range (0, 1)
        grayscale += 1.0
        grayscale *= 0.5
        
        return grayscale

    def texture_shade(self, detail, callback=lambda n, d: None):
        # values used to determine default contrast as a function of detail:

        # SCALING is an overall contrast factor to be applied;
        # useful range around 0.25 to 1.0?
        SCALING = 0.50

        # STEEPNESS determines how rapidly to reduce the contrast
        # as detail increases; useful range around 0.25 to 2.0?
        STEEPNESS = 1.00

        # convert detail from percent to raw value
        detail = detail / 100.0

        if detail == self._prev_detail:
            return

        # "forget" previous value before overwriting its data
        self._prev_detail = None

        callback(0, 100)

        sub_callback = SubProgress(callback, 0, 40)

        if self._fourier is None:
            texture = np.empty_like(self.elevation)
            fourier = np.empty_like(self.elevation)

            fourier[...] = self.elevation

            dct2(fourier, type=2, transpose=True, callback=sub_callback)

            # only save fourier as attribute when computation is complete
            self._fourier = fourier
        else:
            # remove texture from attribute until new computation is complete
            texture = self.texture
            self.texture = None

        callback(40, 100)

        sub_callback = SubProgress(callback, 40, 50)

        half_range = 0.5 * (self.max_elev - self.min_elev)

        normalizer = math.pow(half_range / STEEPNESS, detail) / half_range

        normalizer *= SCALING

        row_factor = self.geo_ref.nrows / float(self.nrows)
        col_factor = self.geo_ref.ncols / float(self.ncols)

        xxscale =  self.distortion.yyres / (col_factor * col_factor)
        xyscale = -self.distortion.xyres / (row_factor * col_factor)
        yyscale =  self.distortion.xxres / (row_factor * row_factor)

        # output includes a factor of __scale ** (detail/2)

        # need to divide by (row_factor * column_factor) ^ (detail/2)

        op = TextureOperator(
            detail, self.ncols, self.nrows, xxscale, yyscale, 2.0 * xyscale)

        texture.shape = self._fourier.shape
        texture[...] = self._fourier

        for j in range(self.ncols):
            op(texture, j, self.nrows)
            sub_callback(j + 1, self.ncols)

        callback(50, 100)

        sub_callback = SubProgress(callback, 50, 90)

        idct2(texture, type=2, transpose=True, callback=sub_callback)

        callback(90, 100)

        sub_callback = SubProgress(callback, 90, 100)

        texture *= normalizer

        self._apply_area_adjustment(texture, detail, sub_callback)

        callback(100, 100)

        # only save texture as attribute when computation is complete
        self.texture = texture

        # save previous detail value once texture is saved
        self._prev_detail = detail

    def _apply_area_adjustment(self, texture, detail, callback):

        pixel_area_sqr = self.distortion.pixel_area_sqr
        sample_rows = self.distortion.sample_rows
        sample_cols = self.distortion.sample_cols

        row_factor = self.geo_ref.nrows / float(self.nrows)
        col_factor = self.geo_ref.ncols / float(self.ncols)

        col_range = np.arange(self.ncols) * col_factor

        if INTERP_AREA_AS_LOG:
            # eliminate any isolated zeros in pixel_area_sqr

            replace_kernel = np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            replace_kernel /= 8.0  # half average of nearest samples

            replace = scipy.ndimage.convolve(
                pixel_area_sqr, replace_kernel, mode='mirror')

            # fill isolated zeros with values derived from neighboring samples

            pixel_area_sqr = np.where(
                pixel_area_sqr == 0.0, replace, pixel_area_sqr)

            replace = None  # release memory

            # interpolate log of pixel area

            pixel_area_spline = scipy.interpolate.RectBivariateSpline(
                sample_rows, sample_cols, np.log(pixel_area_sqr), kx=1, ky=1
            )

            # apply adjustment to output_array
            # (Note: this will extrapolate if geo_ref has fewer rows or cols)

            # process by rows to reduce memory usage
            interp_log_area = np.empty([self.ncols])
            multiplier      = np.empty([self.ncols])
            for i in range(self.nrows):
                # interpolate logs of pixel areas for this row
                interp_log_area[:] = pixel_area_spline(
                    i * row_factor, col_range)

                # determine adjustment factors based on each pixel area
                multiplier[:] = np.exp(interp_log_area * (-0.25 * detail))

                # apply adjustment for projection areal distortion
                texture[i] *= multiplier

                callback(i + 1, self.nrows)

        else:
            # interpolate pixel area

            pixel_area_spline = scipy.interpolate.RectBivariateSpline(
                sample_rows, sample_cols, pixel_area_sqr,
                kx=1, ky=1  # linear interpolation ensures no negative values
            )

            # apply adjustment to output_array
            # (Note: this will extrapolate if geo_ref has fewer rows or cols)

            # ignore infinity from power() below in case pixel area is zero
            with np.errstate(divide='ignore'):
                # process by rows to reduce memory usage
                interp_area_sqr = np.empty([self.ncols])
                multiplier      = np.empty([self.ncols])
                for i in range(self.nrows):
                    # interpolate (squares of) pixel areas for this row
                    interp_area_sqr[:] = pixel_area_spline(
                        i * row_factor, col_range)

                    # determine adjustment factors based on each pixel area
                    multiplier[:] = np.power(interp_area_sqr, -0.25 * detail)

                    # replace (hopefully isolated) infinities with zeros
                    multiplier[interp_area_sqr == 0.0] = 0.0

                    # apply adjustment for projection areal distortion
                    texture[i] *= multiplier

                    callback(i + 1, self.nrows)


class TerrainData():
    def __init__(self, elev_dataset):  # expects gdal.Dataset object

        # Get and validate raster size

        ncols = elev_dataset.RasterXSize
        nrows = elev_dataset.RasterYSize

        if ncols <= 1 or nrows <= 1:
            raise RuntimeError(
                "Elevation dataset empty or only one pixel wide")

        #if self.ncols * self.nrows * 4 > sys.maxsize:
        #    # this is a Pillow limit on image size in bytes
        #    # (see Pillow/src/map.c)
        #    raise RuntimeError(
        #        "Elevation dataset too large to generate image")

        # Allocate space for elevation data

        elev_array = np.empty([nrows, ncols], dtype=np.float32)

        # Read georeference info

        self.geo_transform = (
            elev_dataset.GetGeoTransform(can_return_null=False)
        )
        if not self.geo_transform:
            raise RuntimeError("No georeference info for input file")

        # Read projection info

        self.proj_str = elev_dataset.GetProjection()  # null string if not provided
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.proj_str)

        if srs.IsGeocentric():
            raise RuntimeError("Incompatible projection type (geocentric)")

        geo_ref = GeoReference(nrows, ncols, self.geo_transform, srs)

        # Read elevation data

        nbands = elev_dataset.RasterCount
        if nbands < 1:
            raise RuntimeError("No raster data found in input file")

        band = elev_dataset.GetRasterBand(1)

        datatype = band.DataType

        if datatype == gdal.GDT_Unknown or datatype > gdal.GDT_Float64:
            # unknown or complex data type
            raise RuntimeError(
                "File uses unsupported data type: " +
                gdal.GetDataTypeName(datatype)
            )

        zscale = band.GetScale()  # always None for EHdr (GridFloat) format

        nodata = band.GetNoDataValue()  # None if value not provided
        nodata = np.float32(nodata)  # round to float32; change None to NaN

        elev_array = band.ReadAsArray(
            0, 0, ncols, nrows, buf_obj=elev_array)

        # Generate preview data (downsampled elevation)

        preview_cols = preview_rows = 2 ** PREVIEW_DENSITY

        # adjust preview size to best fit aspect ratio of dataset
        cols_to_rows = (ncols - 1) // nrows
        rows_to_cols = (nrows - 1) // ncols
        while cols_to_rows >= 2 and preview_rows > 4:
            preview_cols *= 2
            preview_rows //= 2
            cols_to_rows //= 4
        while rows_to_cols >= 2 and preview_cols > 4:
            preview_cols //= 2
            preview_rows *= 2
            rows_to_cols //= 4

        if (nrows < 3 * preview_rows or
            ncols < 3 * preview_cols
        ):
            # avoid sampling artifacts on smaller rasters
            resample = gdal.GRIORA_Bilinear
        else:
            # use faster algorithm for larger rasters
            resample = gdal.GRIORA_NearestNeighbour

        try:
            preview_array = band.ReadAsArray(
                0, 0, ncols, nrows,
                preview_cols, preview_rows,
                buf_type=gdal.GDT_Float32, resample_alg=resample
            )
        except TypeError:
            # older versions of GDAL don't give choice of resampling algorithm
            # or choice of data type (except by passing buf_obj argument)
            preview_array = np.empty(
                [preview_rows, preview_cols], dtype=np.float32)
            preview_array = band.ReadAsArray(
                0, 0, ncols, nrows,
                preview_cols, preview_rows,
                buf_obj=preview_array
            )

        band = None

        has_voids = False
        for i in range(nrows):  # process by rows to reduce memory usage
            row = elev_array[i]
            row[row == nodata] = np.nan  # replace NODATA values with NaNs
            has_voids = has_voids or any(np.isnan(row))

        if has_voids:
            preview_array[preview_array == nodata] = np.nan

        if not zscale:
            # assume meters if vertical scale is not provided or zero
            # Warn: "Assuming meters for vertical units of elevation data"
            pass
        elif zscale != 1.0:
            # convert vertical units to meters
            elev_array    *= zscale
            preview_array *= zscale

        # preview_array also within this range if using
        # nearest or bilinear resampling
        min_elev = np.nanmin(elev_array)
        max_elev = np.nanmax(elev_array)

        self.full = TerrainRaster(
            elev_array, geo_ref, min_elev, max_elev, has_voids)
        self.preview = TerrainRaster(
            preview_array, geo_ref, min_elev, max_elev, has_voids)
