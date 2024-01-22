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

import numpy as np

import math


class GeoReference:

    def __init__(self, nrows, ncols, geotransform, srs):
        self.nrows = nrows
        self.ncols = ncols

        self._x1     = geotransform[0]  # outer corner of pixel [0,0]
        self._xstep  = geotransform[1]  # delta_x for each column
        self._xshear = geotransform[2]  # delta_x for each row ?? (normally 0)

        self._y1     = geotransform[3]  # outer corner of pixel [0,0]
        self._yshear = geotransform[4]  # delta_y for each col ?? (normally 0)
        self._ystep  = geotransform[5]  # ystep<0 if raster rows in N->S order

        self._signed_pixel_area = (
            self._xstep * self._ystep - self._xshear * self._yshear)

        if srs.IsProjected() or srs.IsGeographic():
            ecef = srs.CloneGeogCS()
            ecef.SetGeocCS()
            ecef.SetLinearUnits(osr.SRS_UL_METER, 1.0)
            fwd = osr.CoordinateTransformation(ecef, srs)
            rev = osr.CoordinateTransformation(srs, ecef)
            if not fwd.this or not rev.this:
                raise RuntimeError("This map projection not implemented")
            self._project = fwd.TransformPoint
            self._unproject = rev.TransformPoint
            self._project_list = fwd.TransformPoints
            self._unproject_list = rev.TransformPoints

        else:
            # for local coordinate systems, treat the earth as flat

            self._unit = srs.GetLinearUnits()  # meters per map unit

            def local_project(x, y, z):
                return x / self._unit, y / self._unit
            def local_unproject(x, y):
                return x * self._unit, y * self._unit, 0.0
            def local_project_list(xyz):
                return xyz[:,:2] / self._unit
            def local_unproject_list(xy):
                return np.insert(xy * self._unit, 2, 0, 1)

            self._project = local_project
            self._unproject = local_unproject
            self._project_list = local_project_list
            self._unproject_list = local_unproject_list

    def pixel_to_mapxy(self, row, col):  # row in [0,nrows], cols in [0,ncols]
        # Transform pixel values to map coordinates
        x = (self._xstep * col + self._xshear * row) + self._x1
        y = (self._ystep * row + self._yshear * col) + self._y1
        return x, y

    def pixel_to_mapxy_array(self, rowcol):
        # Transform pixel values to map coordinates
        dxy = np.empty(rowcol.shape)
        dxy[:,0] = self._xstep * rowcol[:,1] + self._xshear * rowcol[:,0]
        dxy[:,1] = self._ystep * rowcol[:,0] + self._yshear * rowcol[:,1]
        return dxy + np.array([self._x1, self._y1])

    def mapxy_to_pixel(self, x, y):
        # Transform map coordinates to pixel values
        dx = x - self._x1
        dy = y - self._y1
        row = (self._xstep * dy - self._yshear * dx) / self._signed_pixel_area
        col = (self._ystep * dx - self._xshear * dy) / self._signed_pixel_area
        return row, col

    def mapxy_to_pixel_array(self, xy):
        # Transform map coordinates to pixel values
        dxy = xy - np.array([self._x1, self._y1])
        rowcol = np.empty(xy.shape)
        rowcol[:,0] = self._xstep * dxy[:,1] - self._yshear * dxy[:,0]
        rowcol[:,1] = self._ystep * dxy[:,0] - self._xshear * dxy[:,1]
        return rowcol / self._signed_pixel_area

    def ecef_to_mapxy(self, x, y, z):  # meters
        # Project geocentric coordinates to map coordinates
        x, y = self._project(x, y, z)[:2]  # can fail if undefined
        return x, y

    def ecef_to_mapxy_array(self, xyz):  # meters
        # Project geocentric coordinates to map coordinates
        # (can fail if undefined)
        xy = np.array(self._project_list(xyz), copy=False)[:,:2]
        return xy

    def mapxy_to_ecef(self, x, y):
        # Unproject map coordinates to geocentric coordinates
        x, y, z = self._unproject(x, y)
        return x, y, z  # meters

    def mapxy_to_ecef_array(self, xy):
        # Unproject map coordinates to geocentric coordinates
        xyz = np.array(self._unproject_list(xy), copy=False)
        return xyz  # meters

    def pixel_to_ecef(self, row, col):  # row in [0,nrows], cols in [0,ncols]
        # Convert pixel coordinates to ECEF meters
        x, y = self.pixel_to_mapxy(row, col)
        return self.mapxy_to_ecef(x, y)

    def pixel_to_ecef_array(self, rowcol):
        # Convert pixel coordinates to ECEF meters
        xy = self.pixel_to_mapxy_array(rowcol)
        return self.mapxy_to_ecef_array(xy)

    def ecef_to_pixel(self, x, y, z):
        # Convert ECEF meters to pixel coordinates
        x, y = self.ecef_to_mapxy(x, y, z)  # can fail if undefined
        return self.mapxy_to_pixel(x, y)

    def ecef_to_pixel_array(self, xyz):
        # Convert ECEF meters to pixel coordinates
        xy = self.ecef_to_mapxy_array(xyz)  # can fail if undefined
        return self.mapxy_to_pixel_array(xy)
