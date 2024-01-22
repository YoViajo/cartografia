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

from constants import VERSION_DATE

from tk_setup import tk

from try_imports import pillow, gdal

from errors import notify_abort

from debug_logging import debug_log

import numpy as np

import os
import platform
import sys
import traceback
import warnings


def gdal_version():
    try:
        return gdal.__version__  # from osgeo.gdal
    except Exception:
        pass
    try:
        return gdal.VersionInfo()
    except Exception:
        return "unknown"


def pillow_version():
    try:
        return pillow.__version__  # not available in older versions
    except Exception:
        pass
    try:
        return pillow.PILLOW_VERSION  # deprecated in newer versions
    except Exception:
        return "unknown"


def tcl_tk_versions():
    try:
        return str(tk.TclVersion) + "/" + str(tk.TkVersion)
    except Exception:
        return "unknown"


def tk_window_system():
    # expects Tk to be initialized
    try:
        return tk._default_root.tk.call("tk", "windowingsystem")
    except Exception:
        return "unknown"


def numpy_version():
    try:
        return np.__version__
    except Exception:
        return "unknown"


def python_info():
    try:
        implementation = platform.python_implementation()
    except Exception:
        implementation = "Python"
    version = platform.python_version()
    # using join() ensures no error here in case split() returns single item
    distro = "".join(sys.version.split("|")[1:2])
    return implementation, version, distro


def os_version():
    # Windows version:
    try:
        winver = platform.win32_ver()
        if winver[0]:
            return "Windows " + " ".join(winver)
    except Exception:
        pass

    # Mac OS version:
    try:
        macver = platform.mac_ver()
        if macver[0]:
            version  = "Mac OS " + macver[0] + " " + macver[2] + " "
            version += " ".join(macver[1])
            return version
    except Exception:
        pass

    # Linux distribution:
    try:
        supported_dists = (
                platform._supported_dists + ('arch', 'mageia', 'system'))
        with warnings.catch_warnings():
            # temporarily suppress deprecation warnings
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            # use deprecated function if available
            linuxver = platform.linux_distribution(
                supported_dists=supported_dists)
        if linuxver[0]:
            return "Linux " + " ".join(linuxver)
    except Exception:
        pass

    return "OS unknown"


def system_config():
    # expects Tk to be initialized

    config  = "Texture Shader version " + VERSION_DATE + "\n"
    # GDAL version:
    config += "GDAL " + gdal_version() + ", "
    # Pillow version:
    config += "Pillow " + pillow_version() + "\n"
    # Tcl/TK versions:
    config += "Tcl/Tk " + tcl_tk_versions() + " "
    # Tk windowing system (x11, win32, or aqua)
    config += "on " + tk_window_system() + ", "
    # NumPy version:
    config += "NumPy " + numpy_version() + "\n"
    # Python implementation, version, distro:
    config += " ".join(python_info()) + "\n"
    # system-release-machine-processor-architecture:
    config += platform.platform(aliased=True) + "\n"
    # Windows / Mac OS / Linux version, if available:
    config += os_version() + "\n"
    return config


from tkinter import simpledialog as tkSimpleDialog
from tkinter import dialog       as tkDialog


class ExceptionDialog(tkSimpleDialog.Dialog):
    def __init__(self, parent, message, detail):
        self.message = message
        self.detail = detail
        tkSimpleDialog.Dialog.__init__(
            self, parent, title="Unexpected Error - Texture Shader")

    def body(self, parent):
        parent.master.config(bg="#ECECEC")
        self.msg = tk.Message(parent, text=self.message, aspect=1000, bg="#ECECEC")  # or width=1000
        self.msg.pack(fill=tk.BOTH)
        self.text = tk.Text(parent, width=60, height=40, wrap=tk.WORD, bg="#ECECEC")
        self.text.insert(tk.END, self.detail)
        self.text.pack()

    def buttonbox(self):
        ok_button = tk.Button(self, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        ok_button.pack(padx=5, pady=5)

        self.bind("<Return>", self.ok)



def exception_abort(exc, val, tb):
    # handle any KeyboardInterrupt outside mainloop as usual
    if exc is KeyboardInterrupt:
        debug_log(1, "KeyboardInterrupt caught inside Tk mainloop.")
        raise SystemExit

    err = "".join(traceback.format_exception(exc, val, tb))

    # strip full path from filenames in traceback
    pos2 = 0
    while True:
        pos1 = err.find('File "', pos2)
        if pos1 < 0:
            break
        pos1 += 6
        pos2 = err.find('"', pos1)
        if pos2 < 0:
            break
        filename = os.path.basename(err[pos1:pos2])
        err = err[:pos1] + filename + err[pos2:]
        pos2 += 1

    message = (
        "An unexpected error occurred.\n"
        "Texture Shader will now exit.\n\n"
        "This may indicate a bug in the program,\n"
        "or a problem with your Python installation."
        # "\n\nTexture Shader will now exit."

        # "\nPossible bugs can be reported by emailing the "
        # "error and system info below to bugs@TextureShading.com. "
        # "Technical support is on a limited volunteer basis and "
        # "cannot be guaranteed."

        # "\nReports of potential bugs are appreciated, though "
        # "technical support is limited. You can email "
        # "bugs@TextureShading.com and include the error and "
        # "system info below."

        # "\nReports of potential bugs or other problems are "
        # "appreciated, though technical support is limited."

        # "\nReports of potential bugs are appreciated, "
        # "though technical support is limited."
    )

    detail = ""
    detail += (
        "\nFeedback can be sent to bugs@TextureShading.com. "
        "Please copy the following error and system info "
        "in your message. "
        "Reports of potential bugs are appreciated, "
        "but technical support is limited.\n\n"
        "----------------------------------------\n\n"
    )

    detail += "- Technical Error Details:\n\n" + err + "\n"
    detail += "- System Configuration:\n\n" + system_config()

    detail += "\n----------------------------------------"


    frame = tk.Frame()  # initialize Tk only if not done already
    root = frame._root()
    if not root.winfo_ismapped():
        root.withdraw()  # hide empty root window
    #root.focus_force()  # optional - bring focus to application

    ExceptionDialog(root, message, detail)
    #tkMessageBox.showerror(title=title, message=message, detail=detail)


    # exit immediately to avoid handling any additional exceptions
    #notify_abort("Unexpected Error", message, detail)

    sys.stderr.write("\n")
    sys.stderr.write(message + "\n")
    sys.stderr.write(detail  + "\n")

    if not exit:
        return
    # shutdown gracefully
    try:
        root.destroy()
    except tk.TclError:
        pass
    # exit with error code, without raising SystemExit exception
    os._exit(1)


    # notify user but don't exit yet
    # notify_abort("Unexpected Error", message, detail, False)
    # exit the mainloop and exit the program
    # tk.Frame()._root().quit()
