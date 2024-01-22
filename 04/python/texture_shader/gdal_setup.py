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

import gdal_errors

try:
    from debug_logging import debug_log       as _debug_log
    from debug_logging import debug_exception as _debug_exception
except Exception:
    def _debug_log(severity, message): pass
    def _debug_exception(severity, message): pass


# import GDAL modules

try:
    # for GDAL 1.7 or later
    import osgeo.gdal as gdal
    import osgeo.osr  as osr
except ImportError as e:
    # check for wrong jpeg package
    gdal_errors._check_jpeg_error(e)
    _debug_exception(1, "Import from osgeo failed.")
    try:
        # for GDAL 1.6 or earlier
        import gdal
        import osr
    except ImportError as e:
        # check for wrong jpeg package
        gdal_errors._check_jpeg_error(e)
        # can't find gdal/osr modules
        raise


import os.path
import subprocess
import sys


def _init_gdal_data():
    # set GDAL_DATA config value needed to evaluate EPSG codes, etc.

    # setup_gdal_data() must be called before calling ExportToPCI() on
    # an uninitialized osr.SpatialReference, or else later calls to
    # TransformPoint() will give weird errors even on unrelated objects

    if gdal.GetConfigOption("GDAL_DATA"):
        return

    _debug_log(1, "GDAL_DATA config option not pre-set.")

    gdal_data_path = None

    #try:
    #    # try running "gdal-config --datadir" to get gdal data path
    #    gdal_config = os.path.join(sys.prefix, "bin", "gdal-config")
    #    proc = subprocess.Popen(
    #        [gdal_config, "--datadir"],
    #        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    #    out = proc.communicate()[0]
    #    if not proc.returncode:
    #        # use returned value
    #        gdal_data_path = out.decode(sys.getdefaultencoding()).rstrip()
    #    else:
    #        debug_log(2, "Return code " + str(proc.returnCode) + " from gdal-config.")
    #except Exception:
    #    debug_exception(1, "Exception running gdal-config.")

    try:
        # try running "gdal-config --datadir" to get gdal data path
        gdal_config = os.path.join(sys.prefix, "bin", "gdal-config")
        out = subprocess.check_output(
            [gdal_config, "--datadir"],
            stderr=subprocess.DEVNULL
        )
        gdal_data_path = out.decode(sys.getdefaultencoding()).rstrip()
    except subprocess.CalledProcessError as e:
        _debug_log(
            2, "Return code " + str(e.returncode) + " from gdal-config.")
    except Exception:
        _debug_exception(1, "Exception running gdal-config.")

    if not gdal_data_path:
        # otherwise guess location and look for file "gcs.csv"

        base_paths = []
        base_paths.append(sys.prefix)
        base_paths.append(os.path.join(sys.prefix, "Library"))
        base_paths.append("/usr/local")
        base_paths.append("/usr")

        path_tails = []
        path_tails.append("gdal")
        path_tails.append(os.path.join("gdal", "data"))
        path_tails.append("epsg_csv")
        path_tails.append("")

        try:
            for base in base_paths:
                for tail in path_tails:
                    test_path = os.path.join(base, "share", tail)
                    test_file = os.path.join(test_path, "gcs.csv")
                    if os.path.isfile(test_file):
                        raise StopIteration
        except StopIteration:
            # GDAL files found
            gdal_data_path = test_path
        else:
            # files not found
            raise gdal_errors.GdalDataError(
                "Can't find path to GDAL data files.")

    gdal.SetConfigOption("GDAL_DATA", gdal_data_path)


_init_gdal_data()
