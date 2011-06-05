#ifndef DASH_TYPE_H
#define DASH_TYPE_H

#include <vector>

namespace kiva
{

    //-----------------------------------------------------------------------
    // line dash type
    //-----------------------------------------------------------------------

    class dash_type
    {
        public:
            double phase;
            std::vector<double> pattern;

            // constructor
            dash_type():  phase(0),pattern(2,0)
            {
            }

            // this forces even length of pattern
            dash_type(double _phase, double* _pattern, int n):
                phase(_phase), pattern(n%2 ? n+1 : n)
            {
                for(int i = 0; i < n; i++)
                    pattern[i] = _pattern[i];
                // for odd length patterns, use the first entry as the
                // last gap size. (this is arbitrary)
                if (n%2)
                    pattern[n] = _pattern[0];
            }
            
            ~dash_type()
            {
            }    


            bool is_solid()
            {
                return (pattern.size() == 2 && pattern[0] == 0.);
            }

			// TODO-PZW: define a copy constructor
    };
}

#endif
