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
