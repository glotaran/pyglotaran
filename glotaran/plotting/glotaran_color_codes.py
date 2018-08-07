from enum import Enum
from cycler import cycler
import matplotlib.colors as colors


def hex_to_rgb(hex_string):
    rgb = colors.hex2color(hex_string)
    return tuple([int(255*x) for x in rgb])


def rgb_to_hex(rgb_tuple):
    return colors.rgb2hex([1.0*x/255 for x in rgb_tuple])


class GlotaranDefaultColorCode(Enum):
    # Name	#Hex
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    green = '#00ff00'
    magenta = '#ff00ff'
    cyan = '#00ffff'
    yellow = '#ffff00'
    green4 = '#008b00'
    orange = '#ff8c00'
    brown = '#964b00'
    grey = '#808080'
    violet = '#9400d3'
    turquoise = '#40e0d0'
    maroon = '#800000'
    indigo = '#4b0082'


class GlotaranLineStyles(Enum):
    solid = '-'
    dashed = '--'
    dotted = '..'
    dashdot = '-.'


def get_glotaran_default_colors():
    return [e.value for e in GlotaranDefaultColorCode]


def get_glotaran_default_colors_cycler():
    return cycler('color', [e.value for e in GlotaranDefaultColorCode])
