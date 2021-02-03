""" Parser for inline CSS in style attributes """


def inlineStyle(styleString):
    if len(styleString) == 0:
        return {}
    styles = styleString.split(";")
    rv = dict(style.split(":") for style in styles if len(style) != 0)
    return rv
