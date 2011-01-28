Kiva/Renderer Tasks
-------------------
DONE * Text rendering right side up and positioned correctly.
DONE * Text colors set correctly
DONE * Circles
DONE [kern] * Ellipse
DONE [kern] * Ellipse arc.
* Line Dashing. [not sure if this is being parsed...]
* gradients
* blending
DONE [kern] * image support
* Font lookup isn't very good.  It seems to fail a lot.
* Correct handling of viewBox and x,y,width,height of <g>, et al.
* Relative units (em, ex, %)
* Markers and symbols

Parsing/Document
----------------
DONE * Bold/Italic handled.
* Strike-through and underline. [do we or does the font handle this?]
* Text alignment/anchoring handled [ this will likely require more parsing]
DONE [kern] * Rework parsing of the path data so that it isn't so slow [different parser?]. (was: s/path data/XML. The XML part uses cElementTree, and isn't slow).
* Add lines from svg to the compiled path in one call using lines instead of line_to.
* Line dash styles.

App
---
* Adding kiva.agg and kiva.quartz versions in same UI to get timing
  and quality comparisons.
* Test using Image comparisons
* Cleaning up code so that both wx and kiva backends can use the
  same Document code.


Architecture
------------

* Sepearation of rendering from document.
  I *think* the Document should just build a tree that represents the
  SVG scene.  We then write renderers (wx, kiva, whatever) that walk
  the tree and render it appropriately.  This should simplify the
  separation of the code and improve the modularity of the various
  pieces.

* It would be interesting if the SVG backend to *kiva* actually used
  this same Document as its internal representation before writing
  out to text...

Kiva Bugs
---------
* Mac requires ints for string styles and weights on fonts.
  I think it should take strings like 'bold', 'italic', etc.

Notes
-----
* Should text color be the stroke or fill color?
        The svg library is using fill color.
        Kiva uses stroke color...  Check into this.
