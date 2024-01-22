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

from tk_setup import tk, tkMessageBox, tkFileDialog, tkFont

from try_imports import pillow, gdal

from terrain import TerrainData

from debug_logging import debug_log, debug_exception

from preferences import WELCOME_COLOR, BKGD_COLOR, SLIDER_TROUGH_COLOR

from constants import LICENSE
from constants import VERSION_DATE, VERSION_NAME
from constants import INTERNET_TIMEOUT
from constants import MIN_PREVIEW_WIDTH, MIN_PREVIEW_HEIGHT
from constants import TEST_VERSION

import errors

from system_versions import exception_abort

import numpy as np

try:
    # try Python 3
    from urllib.request import urlopen
except ImportError:
    # Python 2
    from urllib2 import urlopen

import math
import os.path


class DefaultFont:  # requires Tk to be initialized first
    _font_dict = None

    def __init__(self):
        if not self._font_dict:
            font_name = "TkDefaultFont"
            DefaultFont._font_dict = tkFont.Font(font=font_name).actual()

    def __getattr__(self, name):
        return self._font_dict[name]


def find_screen_size(window):
    # save original window state and title
    saved_state = window.state()
    saved_title = window.title()

    # save original requested geometry
    window.update_idletasks()  # update geometry values
    win_width  = window.winfo_reqwidth()
    win_height = window.winfo_reqheight()
    win_x = window.winfo_x()
    win_y = window.winfo_y()
    saved_geom = "{0}x{1}+{2}+{3}".format(
        win_width, win_height, win_x, win_y)

    try:
        # size window to full screen and get size
        window.attributes('-fullscreen', True)  # removes title (why?)
        window.withdraw()  # don't actually draw fullscreen window
        window.update_idletasks()  # update geometry values
        screen_width  = window.winfo_width()
        screen_height = window.winfo_height()
    except tk.TclError:
        debug_log(2, "Fullscreen attribute failed.")
        screen_width  = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
    else:
        # restore window to original state
        window.attributes('-fullscreen', False)
        window.geometry(saved_geom)
        window.title(saved_title)
        window.state(saved_state)

    return screen_width, screen_height


class BorderedCanvas(tk.Canvas):
    def excess_width(self):
        # same as self.winfo_reqwidth() - configured_width
        return 2 * (
                int(self['bd']) +
                int(self['highlightthickness'])
        )

    def excess_height(self):
        # same as self.winfo_reqheight() - configured_height
        return 2 * (
                int(self['bd']) +
                int(self['highlightthickness'])
        )


class BorderedFrame(tk.Frame):
    def excess_width(self):
        return 2 * (
            int(self['padx'].string) +
            int(self['bd']) +
            int(self['highlightthickness'])
        )

    def excess_height(self):
        return 2 * (
            int(self['pady'].string) +
            int(self['bd']) +
            int(self['highlightthickness'])
        )


def maybe_exit():
    raise SystemExit


def test_error():
    raise RuntimeError("Error Test Message")


class Button(tk.Label):
    # replacement for tk.Button, which does not work correctly on Mac OS

    def __init__(self, parent, command, **kw):
        if 'relief' not in kw:
            kw['relief'] = tk.RAISED
        if 'bg' not in kw and 'background' not in kw:
            kw['bg'] = "#ECECEC"
        if 'activebackground' not in kw:
            kw['activebackground'] = "#D3D3D3"  # lightgray
        if 'bd' not in kw and 'borderwidth' not in kw:
            kw['bd'] = 2
        tk.Label.__init__(self, parent, **kw)

        self.default_bkgd = self['background']
        self.pressed_bkgd = self['activebackground']

        self.bind("<Enter>", self.hover_on)
        self.bind("<Leave>", self.hover_off)
        self.bind("<FocusIn>", self.focus_on)
        self.bind("<FocusOut>", self.focus_off)
        self.bind("<Button-1>", self.press)
        self.bind("<ButtonRelease-1>", self.release)
        self.bind("<Return>", self.go)
        self.command = command
        #self["command"] = command
        self.hover_state = False
        self.press_state = False
        self.running = False

    def hover_on(self, event):
        if self.running:
            return
        self.hover_state = True
        if self.press_state:
            self['background'] = self.pressed_bkgd
        else:
            self['background'] = self.default_bkgd

    def hover_off(self, event):
        self.hover_state = False
        # self['highlightbackground'] = self.default_bkgd
        self['background'] = self.default_bkgd

    def focus_on(self, event):
        if self.running:
            return
        #self['highlightbackground'] = self['activebackground']
        self['background'] = self.pressed_bkgd

    def focus_off(self, event):
        self.hover_state = False
        self['background'] = self.default_bkgd

    def press(self, event):
        self.press_state = True
        self['background'] = self.pressed_bkgd

    def release(self, event):
        self.press_state = False
        if not self.hover_state:
            return
        self['background'] = self.default_bkgd
        self.update_idletasks()  # update button color
        if self.running:
            return
        self.running = True
        self.command(event)
        self.running = False

    def go(self, event):
        if self.running:
            return
        self.running = True
        self.command(event)
        self.running = False

    def disable(self):
        pass

    def enable(self):
        pass


class MenuBar(tk.Menu):
    def __init__(self, parent, window):
        #window.option_add('*tearoff', False)  # deoesn't seem to work on Windows

        tk.Menu.__init__(self, window)

        aqua = (window.tk.call("tk", "windowingsystem") == "aqua")

        if aqua:
            app_menu = tk.Menu(self, tearoff=False)
            self.add_cascade(label="Texture Shader", menu=app_menu)
            app_menu.add_command(
                label="About Texture Shader", command=parent.about)
            app_menu.add_command(
                label="License", command=parent.show_license)
            app_menu.add_separator()
            app_menu.add_command(
                label="Quit Texture Shader...", command=maybe_exit)
            self.about_menu = app_menu

        if not aqua:
            file_menu = tk.Menu(self, tearoff=False)
            self.add_cascade(label="File", menu=file_menu)
            file_menu.add_separator()
            file_menu.add_command(label="Quit Texture Shader", command=maybe_exit)

        if TEST_VERSION:
            test_menu = tk.Menu(self, tearoff=False)
            self.add_cascade(label="Test", menu=test_menu)
            test_menu.add_command(
                label="Pillow Error", command=errors.pillow_abort)
            test_menu.add_command(
                label="JPEG Error", command=errors.jpeg_abort)
            test_menu.add_command(
                label="GDAL Error", command=errors.gdal_abort)
            test_menu.add_command(
                label="GDAL_DATA Error", command=errors.gdal_data_abort)
            test_menu.add_command(
                label="Misc. Error", command=test_error)

        if aqua:
            window_menu = tk.Menu(self, name='window', tearoff=False)
            self.add_cascade(menu=window_menu, label='Window')

        if not aqua:
            help_menu = tk.Menu(self, tearoff=False)
            self.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="About Texture Shader", command=parent.about)
            help_menu.add_command(label="License", command=parent.show_license)
            self.about_menu = help_menu

        window.config(menu=self)


class SlidersFrame(tk.Frame):
    def __init__(self, parent, detail_callback, appearance_callback):
        bg = parent['bg']

        tk.Frame.__init__(self, parent, pady=16, bg=bg)

        self.detail_update = detail_callback
        self.appearance_update = appearance_callback

        self.detail     = tk.IntVar()
        self.contrast   = tk.IntVar()
        self.brightness = tk.IntVar()

        SLIDERS_LENGTH = 232  # pixels

        detail_label     = tk.Label(self, text="Texture Detail", bg=bg)
        contrast_label   = tk.Label(self, text="Midtone Contrast", bg=bg)
        brightness_label = tk.Label(self, text="Midtone Brightness", bg=bg)

        detail_label.grid(row=0, column=0, sticky=tk.E)
        contrast_label.grid(row=1, column=0, sticky=tk.E)
        brightness_label.grid(row=2, column=0, sticky=tk.E)

        # use round_detail() instead of resolution=5 for detail
        # slider to avoid conflict with detail spinbox
        detail_slider = tk.Scale(
            self, orient=tk.HORIZONTAL, showvalue=False,
            length=SLIDERS_LENGTH, from_=0, to=200, resolution=5,
            variable=self.detail, command=self.round_detail,
            troughcolor=SLIDER_TROUGH_COLOR
        )
        contrast_slider = tk.Scale(
            self, orient=tk.HORIZONTAL, showvalue=False,
            length=SLIDERS_LENGTH, from_=-40, to=40, resolution=1,
            variable=self.contrast,
            troughcolor=SLIDER_TROUGH_COLOR
        )
        brightness_slider = tk.Scale(
            self, orient=tk.HORIZONTAL, showvalue=False,
            length=SLIDERS_LENGTH, from_=-20, to=20, resolution=1,
            variable=self.brightness,
            troughcolor=SLIDER_TROUGH_COLOR
        )

        detail_slider.grid(row=0, column=1)
        contrast_slider.grid(row=1, column=1)
        brightness_slider.grid(row=2, column=1)

        try:
            # buttonbackground option does not seem to work on Mac OS,
            # and bg sets the text background also, so workaround is
            # to draw an Entry overlaid on top of each Spinbox
            detail_arrows = tk.Spinbox(
                self, width=3, increment=5, from_=0, to=200,
                textvariable=self.detail, bg=bg, takefocus=False,
                highlightbackground=bg, buttonbackground=bg
            )
            contrast_arrows = tk.Spinbox(
                self, width=3, increment=1, from_=-20, to=20,
                textvariable=self.contrast, bg=bg, takefocus=False,
                highlightbackground=bg, buttonbackground=bg
            )
            brightness_arrows = tk.Spinbox(
                self, width=3, increment=1, from_=-20, to=20,
                textvariable=self.brightness, bg=bg, takefocus=False,
                highlightbackground=bg, buttonbackground=bg
            )
        except tk.TclError:
            # Spinbox not available in old versions of Tk
            debug_log(2, "Spinbox creation failed.")
        else:
            detail_arrows.grid(row=0, column=2, sticky=tk.W)
            contrast_arrows.grid(row=1, column=2, sticky=tk.W)
            brightness_arrows.grid(row=2, column=2, sticky=tk.W)

        # from/to limits will be enforced automatically by the sliders
        detail_entry = tk.Entry(
            self, width=3, textvariable=self.detail,
            highlightbackground=bg
        )
        contrast_entry = tk.Entry(
            self, width=3, textvariable=self.contrast,
            highlightbackground=bg
        )
        brightness_entry = tk.Entry(
            self, width=3, textvariable=self.brightness,
            highlightbackground=bg
        )

        detail_entry.grid(row=0, column=2, sticky=tk.W)
        contrast_entry.grid(row=1, column=2, sticky=tk.W)
        brightness_entry.grid(row=2, column=2, sticky=tk.W)

        reset_button = Button(
            self, command=self.reset_defaults, text="Defaults")
        reset_button.grid(row=0, rowspan=3, column=3, padx=10)

        self.detail_trace = self.detail.trace(
            'w', self.detail_update)
        self.contrast_trace = self.contrast.trace(
            'w', self.appearance_update)
        self.brightness_trace = self.brightness.trace(
            'w', self.appearance_update)

        self.reset_defaults()

    def round_detail(self, detail):
        detail = int(detail)
        detail = (detail + 2) // 5 * 5  # round to nearest 5
        self.detail.set(detail)

    def reset_defaults(self, event=None):
        # avoid redraw for contrast & brightness reset
        self.contrast.trace_vdelete('w', self.contrast_trace)
        self.brightness.trace_vdelete('w', self.brightness_trace)

        self.contrast.set(0)
        self.brightness.set(0)

        self.contrast_trace = self.contrast.trace(
            'w', self.appearance_update)
        self.brightness_trace = self.brightness.trace(
            'w', self.appearance_update)

        self.detail.set(50)


class WelcomePanel(tk.Frame):
    def __init__(self, parent):
        margin_color = "#ECECEC"

        bg = WELCOME_COLOR

        tk.Frame.__init__(self, parent, bg=margin_color)

        above = tk.Frame(self, bg=margin_color)
        above.pack(expand=1, fill=tk.Y)

        self.box = tk.Frame(
            self, padx=64, pady=32, bg=bg,
            highlightthickness=1, highlightbackground="black", highlightcolor="black"
        )
        self.box.pack()

        below1 = tk.Frame(self, bg=margin_color)
        below1.pack(expand=1, fill=tk.Y)

        below2 = tk.Frame(self, bg=margin_color)
        below2.pack(expand=1, fill=tk.Y)

        title_font = tkFont.Font(family=DefaultFont().family, size=24)
        title_msg = tk.Message(
            self.box,
            text="Texture Shader in Python",
            justify=tk.CENTER, aspect=2000,
            font=title_font, bg=self.box['bg']
        )
        title_msg.pack(side=tk.TOP, fill=tk.Y)

        version_font = tkFont.Font(family=DefaultFont().family, size=16)
        version_msg = tk.Message(
            self.box,
            text="\n" + VERSION_NAME,
            justify=tk.CENTER, aspect=2000,
            font=version_font, bg=self.box['bg']
        )
        version_msg.pack(side=tk.TOP, fill=tk.Y)

        self.updates_frame = tk.Frame(self.box, bg=self.box['bg'])
        self.updates_frame.pack(side=tk.TOP)

        self.updates_button = Button(
            self.updates_frame, command=self.check_for_updates,
            text="Check for Updates", padx=2, pady=2)
        self.updates_button.pack(side=tk.TOP)

        #license_frame = tk.Frame(
        #    self.box, bg=self.box['bg'],
        #    highlightbackground="#808080", highlightcolor="#808080", highlightthickness=1
        #)
        #license_frame.pack(side=tk.TOP, pady=20)

        #license_button = Button(
        #    license_frame, command=self.show_license,
        #    text="View License", padx=0, pady=0, bd=2,
        #    bg=self.box['bg'], activebackground="palegreen3", relief=tk.FLAT
        #)
        #license_button.pack()

        directions_font = tkFont.Font(family=DefaultFont().family, size=15)
        directions_msg = tk.Message(
            self.box,
            text='\n\nTo get started, press "Open File" to '
                "load a\nDigital Elevation Model (DEM)\n\n",
            justify=tk.CENTER, width=400,
            font=directions_font, bg=self.box['bg']
        )
        directions_msg.pack(side=tk.TOP, fill=tk.Y)

        #quit_frame = tk.Frame(
        #    self.box, bg=self.box['bg'],
        #    highlightbackground="#808080", highlightcolor="#808080", highlightthickness=2
        #)
        #quit_frame.pack(side=tk.LEFT)

        #quit_button = Button(
        #    quit_frame, command=parent.exit, text="      Quit      ", padx=4, pady=4, takefocus=True, bd=2,
        #    bg=self.box['bg'], activebackground="palegreen3", relief=tk.FLAT
        #)
        #quit_button.pack()

        quit_button = Button(
            self.box, command=parent.exit, text="      Quit      ", padx=4, pady=4, takefocus=True
        )
        quit_button.pack(side=tk.LEFT)

        open_button = Button(
            self.box, command=parent.open_elev_file, text="Open File...", padx=4, pady=4, takefocus=True)
        open_button.pack(side=tk.RIGHT)

    def check_for_updates(self, event=None):
        url = (
            "http://www.textureshading.com/"
            "python_texture_shader_version.txt"
        )
        latest_version = ""
        version_info = ""
        try:
            response = urlopen(url, timeout=INTERNET_TIMEOUT)
            latest_version = response.readline(80).decode('utf-8').rstrip()
            version_info   = response.readline(120).decode('utf-8').rstrip()
        except Exception:
            debug_exception(2, "Exception checking latest version.")

        if not latest_version:
            result = "Check for updates failed. Try again later."
            version_info = ""
        elif VERSION_DATE >= latest_version:
            result = "Your current version is up to date."
            version_info = ""
        else:
            result = "A newer version is available to download."

        if version_info:
            result += "\n" + version_info

        self.updates_button.pack_forget()

        updates_msg = tk.Message(
            self.updates_frame,
            text=result,
            justify=tk.CENTER, width=400,
            bg=self.box['bg'])
        updates_msg.pack(side=tk.TOP, fill=tk.Y)

    def show_license(self, event=None):
        tkMessageBox.showinfo(
            title="Texture Shader Software License",
            message=
            "Open-Source License and Disclaimer:",
            detail="".join(LICENSE)
        )
        # return focus to application after message window closes
        self.focus_force()


class PreviewPanel(tk.Frame):
    def __init__(self, parent):
        bg = parent['bg']

        tk.Frame.__init__(self, parent, bg=bg)

        self.main = parent

        self.preview_image = None

        title_font = tkFont.Font(family=DefaultFont().family, size=20)
        title_label = tk.Label(
            self, text="Texture Shading Preview Image", bg=bg, pady=10, font=title_font)
        title_label.grid(row=1, column=1)

        self.preview_frame = BorderedFrame(self, bg=bg)

        self.preview_frame.grid(row=2, column=1, sticky=tk.NSEW)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        # self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(1, weight=1)

        init_width = 800
        init_height = 600
        self.preview_canvas = BorderedCanvas(
            self.preview_frame, relief=tk.SUNKEN, bd=6, highlightthickness=0,
            width=init_width, height=init_height
        )

        # adjust canvas coordinates for borders
        self.preview_canvas.xview_moveto(0.0)
        self.preview_canvas.yview_moveto(0.0)

        # self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.preview_canvas.grid()

        #self.preview_canvas.bind("<Configure>", self.preview_canvas_resize)

        self.preview_frame.bind("<Configure>", self.show_preview)

        self.sliders = SlidersFrame(self, self.new_preview, self.update_preview)
        self.sliders.grid(row=4, column=1)

        directions_font = tkFont.Font(family=DefaultFont().family, size=15)
        directions_msg = tk.Message(
            self,
            text=
                "Use the sliders to adjust the preview image.\n\n"
                "When you're happy with the result, press the Render button "
                "to generate the full-size image.",
            justify=tk.LEFT, aspect=200,
            font=directions_font, bg=bg
        )
        directions_msg.grid(row=2, column=2, padx=30, sticky=tk.S)

        render_button = Button(
            self, command=parent.render_image, text="Render Full Image", padx=4, pady=4, takefocus=True)
        render_button.grid(row=4, column=2, padx=40)

    def reset(self):
        # reset slider controls when new image is loaded
        self.sliders.reset_defaults()

    def new_preview(self, *args):
        if not self.main.terrain:
            return

        try:
            detail = int(self.sliders.detail.get())
        except tk.TclError:
            return

        if detail < 0 or detail > 200:
            detail = max(0, detail)
            detail = min(detail, 200)
            self.sliders.detail.set(detail)

        self.main.terrain.preview.texture_shade(detail)

        self.update_preview()

    def update_preview(self, *args):
        if not self.main.terrain:
            return

        try:
            contrast   = self.sliders.contrast.get()
            brightness = self.sliders.brightness.get()
        except tk.TclError:
            return

        preview_gray = self.main.terrain.preview.grayscale(
            contrast, brightness)

        if preview_gray is None:
            return

        # convert to range (0.5, 255.5)
        preview_gray *= 255.0
        preview_gray += 0.5

        # set any NaN values to black
        preview_gray[np.isnan(preview_gray)] = 0.0

        # convert to integer range [0, 255]
        preview_gray = preview_gray.astype(np.uint8)

        # create Pillow Image sharing memory with preview_gray array
        self.preview_image = pillow.Image.fromarray(preview_gray, "L")
        # self.preview_image = pillow.Image.frombuffer(
        #    "L", (preview_cols, preview_rows), preview_gray, "raw", "L", 0, -1)

        self.show_preview()

    def show_preview(self, event=None):
        canvas = self.preview_canvas
        frame  = self.preview_frame

        width  = frame.winfo_width()  - frame.excess_width()
        height = frame.winfo_height() - frame.excess_height()

        width  -= canvas.excess_width()
        height -= canvas.excess_height()

        if not self.preview_image:
            canvas.config(width=width, height=height)
            return

        min_width  = MIN_PREVIEW_WIDTH
        min_height = MIN_PREVIEW_HEIGHT

        image_width  = self.main.terrain.full.ncols
        image_height = self.main.terrain.full.nrows

        mwih = min_width  * image_height
        mhiw = min_height * image_width
        if mwih >= mhiw:  # min_width is stronger constraint
            # round to nearest int; min_height never reduced here
            min_height = (2 * mwih + image_width) // (2 * image_width)
        else:  # min_height is stronger constraint
            # round to nearest int; min_width never reduced here
            min_width = (2 * mhiw + image_height) // (2 * image_height)

        if not width or not height:
            # width  = int(canvas.cget('width'))
            # height = int(canvas.cget('height'))

            # width  = int(canvas['width'])
            # height = int(canvas['height'])

            width  = canvas.winfo_reqwidth()  - canvas.excess_width()
            height = canvas.winfo_reqheight() - canvas.excess_height()

        wih = width  * image_height
        hiw = height * image_width

        full_width  = width
        full_height = height
        if wih >= hiw:  # image size limited by available height
            # round to nearest int; full_width never increased here
            full_width = (2 * hiw + image_height) // (2 * image_height)
        else:  # image size limited by available width
            # round to nearest int; full_height never increased here
            full_height = (2 * wih + image_width) // (2 * image_width)
        if full_width < min_width or full_height < min_height:
            full_width = min_width
            full_height = min_height
            if width > min_width:
                width = min_width
            elif height > min_height:
                height = min_height
        else:
            width  = full_width
            height = full_height

        scroll_vert = (full_height > height)
        scroll_horz = (full_width  > width)

        # generate resized preview image
        canvas_image = self.preview_image.resize(
            (full_width, full_height), pillow.Image.BICUBIC)

        # hide canvas while resizing and replacing image
        canvas.grid_remove()

        # remove any previous image
        canvas.delete(tk.ALL)
        canvas.image = None  # no longer need reference to old image

        # resize canvas
        canvas.config(width=width, height=height)
        # canvas.grid(side=tk.LEFT, fill=tk.BOTH, expand=1)
        # canvas.pack(side=tk.LEFT, fill=tk.NONE, expand=0)
        # canvas.grid()

        # add new image
        tk_image = pillow.ImageTk.PhotoImage(canvas_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image  # keep a reference to new tk_image

        # unhide canvas
        canvas.grid()

    def preview_canvas_resize(self, event=None):
        pass

    def preview_frame_resize(self, event=None):
        pass


class TextureGUI(tk.Frame):
    def __init__(self, window):
        self.window = window

        self._current_panel = None
        self.terrain = None
        self.texture_gray = None

        screen_width, screen_height = find_screen_size(window)

        border  = window.winfo_rootx() - window.winfo_x()
        top_bar = window.winfo_rooty() - window.winfo_y()

        window_width  = screen_width  * 3 // 4
        window_height = screen_height * 3 // 4

        window_x = (screen_width  - window_width) // 2 - border
        window_y = (screen_height - window_height - top_bar - border) // 2

        window.withdraw()  # hide window until resized
        window.geometry("{0}x{1}+{2}+{3}".format(
            window_width, window_height, window_x, window_y))
        window.minsize(480, 360)
        window.deiconify()

        window.title("Texture Shader")

        window.protocol('WM_DELETE_WINDOW', maybe_exit)
        window.report_callback_exception = exception_abort

        self.menus = MenuBar(self, window)

        tk.Frame.__init__(self, window, bg=BKGD_COLOR)
        self.pack(expand=1, fill=tk.BOTH)

        self.welcome_panel = WelcomePanel(self)

        self.preview_panel = PreviewPanel(self)

        self.switch_to_panel(self.welcome_panel)

    def exit(self, event=None):
        self.window.quit()

    def switch_to_panel(self, panel):
        if self._current_panel:
            self._current_panel.pack_forget()
        self._current_panel = panel
        panel.pack(expand=1, fill=tk.BOTH)

    def open_elev_file(self, event=None):
        # temp window implements workaround for double-click bug on Windows
        temp = tk.Toplevel(self)
        temp.withdraw()
        temp.grab_set()

        elev_path = tkFileDialog.askopenfilename(
            parent=self, title="Choose Elevation File")

        temp.grab_release()  # generate returning <Enter> event before update
        temp.update()  # drain event queue to consume any final ButtonRelease
        temp.destroy()

        if not elev_path:
            # no file chosen
            return

        # Open input file

        try:
            new_dataset = gdal.OpenEx(
                elev_path, gdal.OF_RASTER)  # defaults to READONLY
            if not new_dataset:
                # default message in case failure with no exception raised
                raise RuntimeError(
                    "Unable to open raster file `" + elev_path + "'\n")
        except RuntimeError as e:
            tkMessageBox.showerror(
                parent=self,
                title="Open Failed",
                message="Open failed:\n\n" + str(e))
            return

        # discard any previous rendered image
        self.texture_gray = None

        # Load terrain raster data

        try:
            self.terrain = TerrainData(new_dataset)
        except RuntimeError as e:
            tkMessageBox.showerror(
                parent=self,
                title="Open Failed",
                message="Open failed:\n\n" + str(e))
            return

        #if not self.terrain:
        #    return

        if self.terrain.full.has_voids:
            tkMessageBox.showerror(
                parent=self,
                title="",
                message=
                    "File contains void pixels (NODATA).\n\n"
                    "This version of Texture Shader cannot process terrain "
                    "regions containing void pixels.\n",
                detail=
                    "To fix this:\n\n"
                    "Using another GIS tool, you can either crop a rectangular "
                    "area of complete data, or fill the voids with interpolated "
                    "data. Then try again with the new file.\n\n"
                    "This issue is expected to be resolved in a future version "
                    "of Texture Shader."
            )
            return

        new_dataset = None  # close input file

        self.preview_panel.reset()

        self.switch_to_panel(self.preview_panel)

        self.preview_panel.new_preview()

    def render_image(self, event=None):
        detail     = int(self.preview_panel.sliders.detail.get())
        contrast   = int(self.preview_panel.sliders.contrast.get())
        brightness = int(self.preview_panel.sliders.brightness.get())

        self.progress_percent = 0

        self.terrain.full.texture_shade(detail, self.render_progress)

        self.texture_gray = self.terrain.full.grayscale(
            contrast, brightness)

        # convert to range (0.5, 255.5)
        self.texture_gray *= 255.0
        self.texture_gray += 0.5

        # convert to integer range [0, 255]
        self.texture_gray = self.texture_gray.astype(np.uint8)

        self.save_image_file()

    def render_progress(self, numer, denom):
        percent = math.floor(100.0 * float(numer) / float(denom))
        if percent <= self.progress_percent:
            return
        self.progress_percent = percent

        return

    def save_image_file(self, event=None):
        if self.texture_gray is None:
            return

        has_voids = self.terrain.full.has_voids
        nrows, ncols = self.texture_gray.shape

        image_formats = [
            # (filetype string, driver short name, extension(s))
            ("GeoTIFF", "GTiff", ".tif .tiff"),
            # ("GeoPDF",    "PDF",         ".pdf"),
            # ("JPEG",      "JPEG",        ".jpg .jpeg"), # QUALITY=75 to 95
            # ("PNG",       "PNG",         ".png"),       # ZLEVEL=0 or 6 to 9
            # ("BMP",       "BMP",         ".bmp"),
            # ("GIF",       "GIF",         ".gif"),
            # ("Other Image Type", "",     "*")
        ]

        filetypes = [(a, c) for (a, b, c) in image_formats]

        drivers_by_type = dict((a, b) for (a, b, c) in image_formats)

        drivers_by_extension = {}
        for (a, b, c) in image_formats:
            for d in c.lower().split(" "):
                drivers_by_extension[d] = b

        type_var = tk.StringVar()

        # asksaveasfilename() hangs on Mac OS if default type is ("*")
        type_var.set("GeoTIFF")

        try:
            image_path = tkFileDialog.asksaveasfilename(
                parent=self,
                # title="* Select File Type Below *",
                title="* Saving As GeoTIFF (.tif) File *",
                defaultextension="",  # default from filetypes even on Windows
                filetypes=filetypes,
                typevariable=type_var
            )
        except tk.TclError:
            debug_log(2, "Save file dialog with typevariable option failed.")
            # use without typevariable option for old versions of Tk
            image_path = tkFileDialog.asksaveasfilename(
                parent=self,
                # title="* Select File Type Below *",
                title="* Saving As GeoTIFF (.tif) File *",
                defaultextension="",  # default from filetypes even on Windows
                filetypes=filetypes
            )
            type_var.set("")  # type will be determined from file extension

        if not image_path:
            return

        driver_name = drivers_by_type.get(type_var.get())
        if not driver_name:
            extension = os.path.splitext(image_path)[1].strip().lower()
            driver_name = drivers_by_extension.get(extension)

        driver = gdal.GetDriverByName(driver_name)
        if not driver:
            raise RuntimeError

        if has_voids:
            nbands = 4
            options = ["PHOTOMETRIC=RGB"]
        else:
            nbands = 1
            options = []  # default PHOTOMETRIC=MINISBLACK

        try:
            output = driver.Create(
                image_path, ncols, nrows, nbands, gdal.GDT_Byte,
                options=options)
        except RuntimeError:
            raise

        output.SetGeoTransform(self.terrain.geo_transform)
        output.SetProjection(self.terrain.proj_str)

        if has_voids:
            valid = np.empty([nrows, ncols], np.uint8)
            np.isnan(self.terrain.full.elevation, valid)
            valid -= 1  # 0 (not NaN) -> 255 (valid), 1 (NaN) -> 0 (masked)

        if has_voids:
            for band in range(1, 4):  # RGB bands
                output.GetRasterBand(band).WriteArray(self.texture_gray)
            output.GetRasterBand(4).WriteArray(valid)  # alpha band
        else:
            outband = output.GetRasterBand(1)
            outband.WriteArray(self.texture_gray)
            # if has_voids:
            #    try:
            #        output.CreateMaskBand(gdal.GMF_PER_DATASET)
            #    except AttributeError:
            #        pass  # old versions of GDAL didn't have mask bands
            #    else:
            #        outband.GetMaskBand().WriteArray(valid)
            outband = None

        valid = None

        output = None  # close the output file


    def about(self):
        tkMessageBox.showinfo(
            title="About Texture Shader",
            message=
                "\nTexture Shader in Python\n\n" +
                VERSION_NAME +
                "\n",
            detail=
                "\nProblem reports, suggestions, and comments \n"
                "are welcome and can be sent to:\n\n"
                "          feedback@TextureShading.com\n\n"
                "Support is on a limited, volunteer basis,\n"
                "however, so a response cannot be guaranteed.\n"
        )
        # return focus to application after message window closes
        self.focus_force()


    def show_license(self):
        tkMessageBox.showinfo(
            title="Texture Shader Software License",
            message=
            "Open-Source License and Disclaimer:",
            detail="".join(LICENSE)
        )
        # return focus to application after message window closes
        self.focus_force()
