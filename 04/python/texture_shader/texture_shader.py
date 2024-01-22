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

from try_imports import pillow, gdal

from tk_setup import tk

from texture_interface import TextureGUI

from debug_logging import debug_log

from system_versions import system_config

import sys


def main():
    root = tk.Tk()  # create Tcl interpreter and main window

    sys.stderr.write("\n")
    sys.stderr.write(system_config())  # expects Tk to be initialized
    sys.stderr.write("\n")

    gdal.UseExceptions()  # also suppresses error messages to stdout

    gdal.SetConfigOption("GDAL_TIFF_INTERNAL_MASK", "YES")

    debug_log(
        1, "GDAL_DATA set to '" + gdal.GetConfigOption("GDAL_DATA") + "'.")

    main = TextureGUI(root)

    tk.mainloop()


if __name__ == "__main__":
    main()

