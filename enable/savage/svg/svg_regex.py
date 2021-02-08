# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Small hand-written recursive descent parser for SVG <path> data.


In [1]: from svg_regex import svg_parser

In [3]: svg_parser.parse('M 10,20 30,40V50 60 70')
Out[3]: [('M', [(10.0, 20.0), (30.0, 40.0)]), ('V', [50.0, 60.0, 70.0])]

In [4]: svg_parser.parse('M 0.6051.5')  # An edge case
Out[4]: [('M', [(0.60509999999999997, 0.5)])]

In [5]: svg_parser.parse('M 100-200')  # Another edge case
Out[5]: [('M', [(100.0, -200.0)])]
"""

import re
from functools import partial


# Sentinel.
class _EOF(object):
    def __repr__(self):
        return "EOF"


EOF = _EOF()

lexicon = [
    (
        "float",
        r"[-\+]?(?:(?:[0-9]*\.[0-9]+)|(?:[0-9]+\.?))(?:[Ee][-\+]?[0-9]+)?",
    ),
    ("int", r"[-\+]?[0-9]+"),
    ("command", r"[AaCcHhLlMmQqSsTtVvZz]"),
]


class Lexer(object):
    """ Break SVG path data into tokens.

    The SVG spec requires that tokens are greedy. This lexer relies on Python's
    regexes defaulting to greediness.

    This style of implementation was inspired by this article:

        http://www.gooli.org/blog/a-simple-lexer-in-python/
    """

    def __init__(self, lexicon):
        self.lexicon = lexicon
        parts = []
        for name, regex in lexicon:
            parts.append("(?P<%s>%s)" % (name, regex))
        self.regex_string = "|".join(parts)
        self.regex = re.compile(self.regex_string)

    def lex(self, text):
        """ Yield (token_type, str_data) tokens.

        The last token will be (EOF, None) where EOF is the singleton object
        defined in this module.
        """
        for match in self.regex.finditer(text):
            for name, _ in self.lexicon:
                m = match.group(name)
                if m is not None:
                    yield (name, m)
                    break
        yield (EOF, None)


svg_lexer = Lexer(lexicon)


class SVGPathParser(object):
    """ Parse SVG <path> data into a list of commands.

    Each distinct command will take the form of a tuple (command, data). The
    `command` is just the character string that starts the command group in the
    <path> data, so 'M' for absolute moveto, 'm' for relative moveto, 'Z' for
    closepath, etc. The kind of data it carries with it depends on the command.
    For 'Z' (closepath), it's just None. The others are lists of individual
    argument groups. Multiple elements in these lists usually mean to repeat
    the command. The notable exception is 'M' (moveto) where only the first
    element is truly a moveto. The remainder are implicit linetos.

    See the SVG documentation for the interpretation of the individual elements
    for each command.

    The main method is `parse(text)`. It can only consume actual strings, not
    filelike objects or iterators.
    """

    def __init__(self, lexer=svg_lexer):
        self.lexer = lexer

        self.command_dispatch = {
            "Z": self.rule_closepath,
            "z": self.rule_closepath,
            "M": self.rule_moveto_or_lineto,
            "m": self.rule_moveto_or_lineto,
            "L": self.rule_moveto_or_lineto,
            "l": self.rule_moveto_or_lineto,
            "H": self.rule_orthogonal_lineto,
            "h": self.rule_orthogonal_lineto,
            "V": self.rule_orthogonal_lineto,
            "v": self.rule_orthogonal_lineto,
            "C": self.rule_curveto3,
            "c": self.rule_curveto3,
            "S": self.rule_curveto2,
            "s": self.rule_curveto2,
            "Q": self.rule_curveto2,
            "q": self.rule_curveto2,
            "T": self.rule_curveto1,
            "t": self.rule_curveto1,
            "A": self.rule_elliptical_arc,
            "a": self.rule_elliptical_arc,
        }

        self.number_tokens = set(["int", "float"])

    def parse(self, text):
        """ Parse a string of SVG <path> data.
        """
        svg_iterator = self.lexer.lex(text)
        token = next(svg_iterator)
        return self.rule_svg_path(partial(next, svg_iterator), token)

    def rule_svg_path(self, next, token):
        commands = []
        while token[0] is not EOF:
            if token[0] != "command":
                raise SyntaxError("expecting a command; got %r" % (token,))
            rule = self.command_dispatch[token[1]]
            command_group, token = rule(next, token)
            commands.append(command_group)
        return commands

    def rule_closepath(self, next, token):
        command = token[1]
        token = next()
        return (command, None), token

    def rule_moveto_or_lineto(self, next, token):
        command = token[1]
        token = next()
        coordinates = []
        while token[0] in self.number_tokens:
            pair, token = self.rule_coordinate_pair(next, token)
            coordinates.append(pair)
        return (command, coordinates), token

    def rule_orthogonal_lineto(self, next, token):
        command = token[1]
        token = next()
        coordinates = []
        while token[0] in self.number_tokens:
            coord, token = self.rule_coordinate(next, token)
            coordinates.append(coord)
        return (command, coordinates), token

    def rule_curveto3(self, next, token):
        command = token[1]
        token = next()
        coordinates = []
        while token[0] in self.number_tokens:
            pair1, token = self.rule_coordinate_pair(next, token)
            pair2, token = self.rule_coordinate_pair(next, token)
            pair3, token = self.rule_coordinate_pair(next, token)
            coordinates.append((pair1, pair2, pair3))
        return (command, coordinates), token

    def rule_curveto2(self, next, token):
        command = token[1]
        token = next()
        coordinates = []
        while token[0] in self.number_tokens:
            pair1, token = self.rule_coordinate_pair(next, token)
            pair2, token = self.rule_coordinate_pair(next, token)
            coordinates.append((pair1, pair2))
        return (command, coordinates), token

    def rule_curveto1(self, next, token):
        command = token[1]
        token = next()
        coordinates = []
        while token[0] in self.number_tokens:
            pair1, token = self.rule_coordinate_pair(next, token)
            coordinates.append(pair1)
        return (command, coordinates), token

    def rule_elliptical_arc(self, next, token):
        command = token[1]
        token = next()
        arguments = []
        while token[0] in self.number_tokens:
            rx = float(token[1])
            if rx < 0.0:
                raise SyntaxError(
                    "expecting a nonnegative number; got %r" % (token,)
                )

            token = next()
            if token[0] not in self.number_tokens:
                raise SyntaxError("expecting a number; got %r" % (token,))
            ry = float(token[1])
            if ry < 0.0:
                raise SyntaxError(
                    "expecting a nonnegative number; got %r" % (token,)
                )

            token = next()
            if token[0] not in self.number_tokens:
                raise SyntaxError("expecting a number; got %r" % (token,))
            axis_rotation = float(token[1])

            token = next()
            if token[1] not in ("0", "1"):
                raise SyntaxError(
                    "expecting a boolean flag; got %r" % (token,)
                )
            large_arc_flag = bool(int(token[1]))

            token = next()
            if token[1] not in ("0", "1"):
                raise SyntaxError(
                    "expecting a boolean flag; got %r" % (token,)
                )
            sweep_flag = bool(int(token[1]))

            token = next()
            if token[0] not in self.number_tokens:
                raise SyntaxError("expecting a number; got %r" % (token,))
            x = float(token[1])

            token = next()
            if token[0] not in self.number_tokens:
                raise SyntaxError("expecting a number; got %r" % (token,))
            y = float(token[1])

            token = next()
            arguments.append(
                ((rx, ry), axis_rotation, large_arc_flag, sweep_flag, (x, y))
            )

        return (command, arguments), token

    def rule_coordinate(self, next, token):
        if token[0] not in self.number_tokens:
            raise SyntaxError("expecting a number; got %r" % (token,))
        x = float(token[1])
        token = next()
        return x, token

    def rule_coordinate_pair(self, next, token):
        # Inline these since this rule is so common.
        if token[0] not in self.number_tokens:
            raise SyntaxError("expecting a number; got %r" % (token,))
        x = float(token[1])
        token = next()
        if token[0] not in self.number_tokens:
            raise SyntaxError("expecting a number; got %r" % (token,))
        y = float(token[1])
        token = next()
        return (x, y), token


svg_parser = SVGPathParser()
