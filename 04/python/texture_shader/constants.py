LICENSE = (
    "Copyright (c) 2019 Leland Brown.\n",
    "All rights reserved.\n\n",

    "Redistribution and use in source and binary forms, with or without ",
    "modification, are permitted provided that the following conditions ",
    "are met:\n\n",

    "1. Redistributions of source code must retain the above copyright ",
    "notice, this list of conditions and the following disclaimer.\n\n",

    "2. Redistributions in binary form must reproduce the above copyright ",
    "notice, this list of conditions and the following disclaimer in the ",
    "documentation and/or other materials provided with the distribution.\n\n"
    ,
    "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ",
    "``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT ",
    "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ",
    "A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT ",
    "HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,",
    " SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT ",
    "LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, ",
    "DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY ",
    "THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT ",
    "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE ",
    "OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
)

VERSION_DATE = "2019-10-15"

VERSION_NAME = "Alpha Version " + VERSION_DATE

MIN_PREVIEW_WIDTH  = 320  # in pixels
MIN_PREVIEW_HEIGHT = 240

# num pixels per axis used to interpolate projection distortion
# NUM_DISTORTION_SAMPLES = 16
NUM_DISTORTION_SAMPLES = 256

INTERNET_TIMEOUT = 5.0  # seconds

USE_CUTOFF_FREQUENCY = False  # False to use actual fractional Laplacian

INTERP_AREA_AS_LOG = False  # pixel area interpolation method

# TERRAIN_OPERATOR_METHOD:
#     0 = model Laplacian using cubic spline surface
#     1 = model infinitesimal fractional Laplacian using cubic spline
TERRAIN_OPERATOR_METHOD = 0

TEST_VERSION = False

MIN_PAGE_SIZE = 2048  # or 4096 or 512? - lower bound on system page size
