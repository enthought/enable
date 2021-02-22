// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include "kiva_gradient.h"

using namespace kiva;

gradient::gradient(gradient_type_e gradient_type) :
    gradient_type(gradient_type),
    spread_method(pad)
{
}

gradient::gradient(gradient_type_e gradient_type, std::vector<point> points,
                    std::vector<gradient_stop> stops, const char* spread_method,
                    const char* units) :
    points(points),
    stops(stops),
    gradient_type(gradient_type),
    spread_method(pad)
{
    if (strcmp(spread_method, "reflect") == 0)
        this->spread_method = kiva::reflect;
    else if (strcmp(spread_method, "repeat") == 0)
        this->spread_method = kiva::repeat;

    if (strcmp(units, "userSpaceOnUse") == 0)
        this->units = kiva::user_space;
    else
		this->units = kiva::object_bounding_box;
}

gradient::~gradient()
{
}
