from __future__ import print_function

import numpy

from kiva.api import points_in_polygon

poly = numpy.array((( 0.0,  0.0),
                    (10.0,  0.0),
                    (10.0, 10.0),
                    ( 0.0, 10.0)))

print(point_in_polygon(-1,-1,poly))
print(point_in_polygon(0.0,0.0,poly))
print(point_in_polygon(5,5,poly))
print(point_in_polygon(10,10,poly))
print(point_in_polygon(15,15,poly))

pts = numpy.array(((-1.0, -1.0),
                   ( 0.1,  0.0),
                   ( 0.0,  0.1),
                   ( 0.0,  0.0),
                   ( 5.0,  5.0),
                   ( 10.0, 10.0),
                   ( 15.0, 15.0)))

results = points_in_polygon(pts, poly)

print(results)

pts = numpy.random.random_sample((20000, 2))*12.5-2.5

import time
t1 = time.clock()
results = points_in_polygon(pts, poly)
t2 = time.clock()
print('points_in_polygon() for %d pts in %d point polygon (sec): %f' % \
      (len(pts), len(poly), t2-t1))
print(pts[:5])
print(results[:5])

poly = numpy.array((( 0.0,  0.0),
                    ( 2.0,  0.0),
                    ( 5.0,  0.0),
                    ( 7.5,  0.0),
                    (10.0,  0.0),
                    (10.0,  2.5),
                    (10.0,  5.0),
                    (10.0,  7.5),
                    (10.0, 10.0),
                    ( 7.5, 10.0),
                    ( 5.0, 10.0),
                    ( 2.5, 10.0),
                    ( 0.0, 10.0)))

t1 = time.clock()
results = points_in_polygon(pts, poly)
t2 = time.clock()
print('points_in_polygon() for %d pts in %d point polygon (sec): %f' % \
      (len(pts), len(poly), t2-t1))
print(pts[:5])
print(results[:5])
