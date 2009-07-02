#include "kiva_gradient.h"

using namespace kiva;

gradient::gradient(gradient_type_e gradient_type) :
    gradient_type(gradient_type),
    spread_method(pad)
{
}

gradient::gradient(gradient_type_e gradient_type, std::vector<point> points,
                    std::vector<gradient_stop> stops, const char* spread_method) :
    points(points),
    stops(stops),
    gradient_type(gradient_type),
    spread_method(pad)
{
    if (spread_method == "reflect")
        this->spread_method = kiva::reflect;
    else if (spread_method == "repeat")
        this->spread_method = kiva::repeat;
}

gradient::~gradient()
{
}
