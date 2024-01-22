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

from tk_setup import tk, tkMessageBox, tkSimpleDialog

import os
import sys


def notify_abort(title, message, detail, exit=True):
    sys.stderr.write("\n")
    sys.stderr.write(message + "\n")
    sys.stderr.write(detail  + "\n")
    title += " - Texture Shader"
    detail += "\n\n- Press OK to exit Texture Shader."
    frame = tk.Frame()  # initialize Tk only if not done already
    root = frame._root()
    if not root.winfo_ismapped():
        root.withdraw()  # hide empty root window
    #root.focus_force()  # optional - bring focus to application
    tkMessageBox.showerror(title=title, message=message, detail=detail)
    if not exit:
        return
    # shutdown gracefully
    try:
        root.destroy()
    except tk.TclError:
        pass
    # exit with error code, without raising SystemExit exception
    os._exit(1)


def pillow_abort():
    message = (
        "Error:\n\n"
        'Package "pillow" (PIL) is not installed in your Python environment\n'
    )
    detail = (
        'Texture Shader requires the "pillow" package\n'
        "(Python Imaging Library).\n\n"
        "- To fix this:\n\n"
        "Try using one of the following commands in a terminal window, "
        "depending on your Python package manager:\n\n"
        "        conda install pillow\n"
        "        pip install pillow\n"
    )
    notify_abort("Missing Python Package", message, detail)


def jpeg_abort(version=0):
    message = (
        "Error:\n\n"
        "An incompatible version of the jpeg library is installed "
        "in your Python environment\n"
    )
    if version:
        detail = (
            "Texture Shader uses the gdal package, " +
            "which requires the jpeg library version {0}.".format(version) +
            "\n\nVersion {0} of jpeg is not compatible.".format(version+1) +
            "\n\n- To fix this:\n\n"
            "Try using one of the following commands in a terminal window, "
            "depending on your Python package manager:\n\n" +
            "        conda install jpeg={0}\n".format(version) +
            "        pip install jpeg={0}\n".format(version)
        )
    else:
        err = str(sys.exc_info()[1])
        detail = (
            "Texture Shader uses the gdal package, " +
            "which may require a different version of "
            "the jpeg library than you have installed.\n\n"
            "Here is the full error message:\n\n" + err
        )

    notify_abort("Incompatible Python Package", message, detail)


def gdal_abort():
    message = (
        "Error:\n\n"
        'Package "gdal" is not installed in your Python environment\n'
    )
    detail = (
        "Texture Shader requires the gdal package\n"
        "(Geospatial Data Abstraction Library).\n\n"
        "- To fix this:\n\n"
        "Try using one of the following commands in a terminal window, "
        "depending on your Python package manager:\n\n"
        "        conda install gdal\n"
        "        pip install gdal\n"
    )
    notify_abort("Missing Python Package", message, detail)


def gdal_data_abort():
    default_gdal_path = os.path.join(sys.prefix, "share", "gdal")
    message = (
        "Error:\n\n"
        "Can't locate GDAL data files\n"
    )
    detail = (
            "To fix this:\n\n"
            "1. Create the following folder if it does not exist:\n\n"
            "        " + default_gdal_path + "\n\n" +
            '2. Find a file named "gcs.csv" on your computer.\n\n'
            "3. Copy all files from the folder containing gcs.csv "
            "into the folder created in step 1.\n\n"
            #"3. Copy gcs.csv and all other files in the same folder "
            #"into the folder created in step 1.\n\n"
            "4. If you cannot find gcs.csv, try one of these commands "
            "in a terminal window:\n\n"
            "        conda install gdal\n"
            "        pip install gdal\n"
    )
    notify_abort("Missing Data Files", message, detail)
