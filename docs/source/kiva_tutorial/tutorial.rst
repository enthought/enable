============
Introduction
============

This tutorial is intended to help you get up and running working with kiva.
Kiva is a backend-toolkit agnostic 2D vector drawing interface.  In other
words, it is a Python interface layer which sits on top of many different
backends which provide 2D vector drawing functionality such as Quartz, Cairo,
etc.  Many of the concepts that will be covered here are generalizations of the
ideas that govern the underlying backends.  As such, new kiva users may find it
useful for their general understanding of what kiva is all about to go through
any of the numerous other tutorials / documentation out there for specific
backends. Here are some we recommend: 



Before we dive in, we recommend at least skimming the kiva documentation before
going through the tutorial so you are familiar with the major terms and concepts.



In this tutorial, we will go through the process of drawing the basic circuit
diagram shown below step by step, with kiva.  As mentioned, kiva supports a
variety of different, but for this tutorial we will work with the default agg (ref to kiva backends page)
backend.

[IMAGE OF FINISHED DIAGRAM HERE]


Starting from the beginning, we will need a GraphicsContext so we import it
from our desired backend and instantiate it.

.. literalinclude:: tutorial.py
    :lines: 2, 5


Now we are ready to use it to start drawing, simple as that.  Lets start with
just drawing the wires.  Given that they are rectangles, this can be done easily
using the graphics context's `rect` method.  






As you may have noticed, most of the code for drawing the Ammeter and the
Voltmeter was effectively the same.  Sometimes it is useful to work with an
independent path instance as opposed to specifically messing with the current
path of the graphics context. This brings us to the notion of CompiledPaths,
which we will now use to draw the resistors. As you can see the path for each
resistor will be exactly the same.

