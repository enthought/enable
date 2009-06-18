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

void gradient::_apply_linear_transform(point p1, point p2, agg::trans_affine& mtx, double d2)
{
    double dx = p2.first - p1.first;
    double dy = p2.second - p1.second;
    mtx.reset();
    mtx *= agg::trans_affine_scaling(sqrt(dx * dx + dy * dy) / d2);
    mtx *= agg::trans_affine_rotation(atan2(dy, dx));
    mtx *= agg::trans_affine_translation(p1.first, p1.second);
    mtx.invert();
}