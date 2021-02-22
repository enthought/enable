// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include "kiva_rect.h"
#include <algorithm>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

namespace kiva
{

rect_type disjoint_intersect(const rect_type &a, const rect_type &b)
{
    double xl = max(a.x, b.x);
    double yb = max(a.y, b.y);
    double xr = min(a.x2(), b.x2());
    double yt = min(a.y2(), b.y2());
    if ((xr >= xl) && (yt >= yb))
    {
        return rect_type(xl, yb, xr-xl, yt-yb);
    }
    else
    {
        return rect_type(xl, yb, -1, -1);
    }
}

rect_list_type disjoint_intersect(const rect_list_type &rects)
{
    if (rects.size() < 2)
    {
        return rects;
    }

    rect_list_type result_list;
    result_list.push_back(rects[0]);
    for (unsigned int i=1; i<rects.size(); i++)
    {
        result_list = disjoint_intersect(result_list, rects[i]);
    }
    return result_list;
}

rect_list_type disjoint_intersect(const rect_list_type &original_list,
                                  const rect_type &new_rect)
{
    rect_list_type result_list;
    if (original_list.size() == 0)
    {
        result_list.push_back(new_rect);
        return result_list;
    }

    rect_type result_rect;
    for (unsigned int i=0; i<original_list.size(); i++)
    {
        result_rect = disjoint_intersect(new_rect, original_list[i]);
        if ((result_rect.w >= 0) && (result_rect.h >= 0))
        {
            result_list.push_back(result_rect);
        }
    }
    
    return result_list;
}


rect_list_type disjoint_union(const rect_type &a, const rect_type &b)
{
    rect_list_type rlist;
    rlist.push_back(a);
    return disjoint_union(rlist, b);
}

rect_list_type disjoint_union(const rect_list_type &rects)
{
    if (rects.size() < 2)
    {
        return rects;
    }

    rect_list_type rlist;
    rlist.push_back(rects[0]);
    for (unsigned int i=1; i<rects.size(); i++)
    {
        rlist = disjoint_union(rlist, rects[i]);
    }
    return rlist;
    
}

rect_list_type disjoint_union(rect_list_type original_list,
                              const rect_type &new_rect)
{

    // short-circuit:
    if (original_list.size() == 0)
    {
        original_list.push_back(new_rect);
        return original_list;
    }

    rect_list_type additional_rects;
    rect_list_type todo;
    todo.push_back(new_rect);
    
    // Iterate over each item in the todo list:
    unsigned int todo_count = 0;
    bool use_leftover = true;
    while (todo_count < todo.size())
    {
        use_leftover = true;

        rect_type *cur_todo = &todo[todo_count];

        double xl1 = cur_todo->x;
        double yb1 = cur_todo->y;
        double xr1 = cur_todo->x2();
        double yt1 = cur_todo->y2();

        double xl2, yb2, xr2, yt2;

        unsigned int orig_count = 0;
        while (orig_count < original_list.size())
        {
            rect_type *cur_orig = &original_list[orig_count];
            xl2 = cur_orig->x;
            yb2 = cur_orig->y;
            xr2 = cur_orig->x2();
            yt2 = cur_orig->y2();

            // Test for non-overlapping
            if ((xl1 >= xr2) || (xr1 <= xl2) || (yb1 >= yt2) || (yt1 <= yb2))
            {
                orig_count++;
                continue;
            }

            // Test for new rect being wholly contained in an existing one
            bool x1inx2 = ((xl1 >= xl2) && (xr1 <= xr2));
            bool y1iny2 = ((yb1 >= yb2) && (yt1 <= yt2));
            if (x1inx2 && y1iny2)
            {
                use_leftover = false;
                break;
            }
    
            // Test for existing rect being wholly contained in new rect
            bool x2inx1 = ((xl2 >= xl1) && (xr2 <= xr1));
            bool y2iny1 = ((yb2 >= yb1) && (yt2 <= yt1));
            if (x2inx1 && y2iny1)
            {
                // Erase the existing rectangle from the original_list
                // and set the iterator to the next one.
                original_list.erase(original_list.begin() + orig_count);
                continue;
            }

            // Test for rect 1 being within rect 2 along the x-axis:
            if (x1inx2)
            {
                if (yb1 < yb2)
                {
                    if (yt1 > yt2)
                    {
                        todo.push_back(rect_type(xl1, yt2, xr1-xl1, yt1-yt2));
                    }
                    yt1 = yb2;
                } else {
                    yb1 = yt2;
                }
                orig_count++;
                continue;
            }

            // Test for rect 2 being within rect 1 along the x-axis:
            if (x2inx1)
            {
                if (yb2 < yb1)
                {
                    if (yt2 > yt1)
                    {
                        original_list.insert(original_list.begin() + orig_count,
                                             rect_type(xl2, yt1, xr2-xl2, yt2-yt1));
                        orig_count++;
                    }
                    original_list[orig_count] = rect_type(xl2, yb2, xr2-xl2, yb1-yb2);
                } else {
                    original_list[orig_count] = rect_type(xl2, yt1, xr2-xl2, yt2-yt1);
                }
                orig_count++;
                continue;
            }

            // Test for rect 1 being within rect 2 along the y-axis:
            if (y1iny2)
            {
                if (xl1 < xl2)
                {
                    if (xr1 > xr2)
                    {
                        todo.push_back(rect_type(xr2, yb1, xr1-xr2, yt1-yb1));
                    }
                    xr1 = xl2;
                } else {
                    xl1 = xr2;
                }
                orig_count++;
                continue;
            }
            
            // Test for rect 2 being within rect 1 along the y-axis:
            if (y2iny1)
            {
                if (xl2 < xl1)
                {
                    if (xr2 > xr1)
                    {
                        original_list.insert(original_list.begin() + orig_count,
                                             rect_type(xr1, yb2, xr2-xr1, yt2-yb2));
                        orig_count++;
                    }
                    original_list[orig_count] = rect_type(xl2, yb2, xl1-xl2, yt2-yb2);
                } else {
                    original_list[orig_count] = rect_type(xr1, yb2, xr2-xr1, yt2-yb2);
                }
                orig_count++;
                continue;
            }

            // Handle a 'corner' overlap of rect 1 and rect 2:
            double xl, yb, xr, yt;
            if (xl1 < xl2)
            {
                xl = xl1;
                xr = xl2;
            } else {
                xl = xr2;
                xr = xr1;
            }
            
            if (yb1 < yb2)
            {
                yb  = yb2;
                yt  = yt1;
                yt1 = yb2;
            } else {
                yb  = yb1;
                yt  = yt2;
                yb1 = yt2;
            }

            todo.push_back(rect_type(xl, yb, xr-xl, yt-yb));
            
            orig_count++;
        }

        if (use_leftover)
        {
            additional_rects.push_back(rect_type(xl1, yb1, xr1-xl1, yt1-yb1));
        }
        todo_count++;
    }

    for (rect_list_type::iterator it = additional_rects.begin(); it != additional_rects.end(); it++)
    {
        original_list.push_back(*it);
    }

    return original_list;
}


bool rect_list_contains(rect_list_type &l, rect_type &r)
{
    return (std::find(l.begin(), l.end(), r) != l.end());
}

// Test code
//
// There are 17 types of rectangle-rectangle intersection:
//      1. no intersections
//      2. new rect is contained by old rect
//      3. new rect surrounds old rect
//      4-7. new rect overlaps one of the four corners
//      8-11. new rect overlaps (but doesn't span) one of the edges
//      12-15. new rect contains one of the existing edges
//      16-17. new rect overlaps (but doesn't span) two opposite edges

// case 1
void test_disjoint_outside()
{
    rect_list_type cliprects;
    rect_type rect1(20,20,40,40);
    rect_type rect2(70,20,40,40);
    cliprects = disjoint_union(rect1, rect2);
    assert(cliprects.size() == 2);
    assert(rect_list_contains(cliprects, rect1));
    assert(rect_list_contains(cliprects, rect2));
}

// cases 2 and 3
void test_disjoint_2_3()
{
    rect_list_type cliprects;
    rect_type bigrect(10,10,80,80);
    rect_type smallrect(15,15,50,10);
    // case 2
    cliprects = disjoint_union(bigrect, smallrect);
    if ((cliprects.size() != 1) || (cliprects[0] != bigrect))
    {
        printf("Error in test_disjoint_2_3(): case 2.\n");
    }
    // case 3
    cliprects = disjoint_union(smallrect, bigrect);
    if ((cliprects.size() != 1) || (cliprects[0] != bigrect))
    {
        printf("Error in test_disjoint_2_3(): case 3.\n");
    }
}

// cases 4-7
void test_disjoint_corner()
{
    bool all_pass = false;
    
    rect_list_type cliprects;
    rect_type mainrect(40,40,20,20);
    rect_type ul(35,55,10,10);
    rect_type ur(55,55,10,10);
    rect_type ll(35,35,10,10);
    rect_type lr(55,35,10,10);

    // upper left
    cliprects = disjoint_union(mainrect, ul);
    rect_type ul_1(35, 55, 5, 5);
    rect_type ul_2(35, 60, 10, 5);
    all_pass = (cliprects.size() == 3) && rect_list_contains(cliprects, ul_1) && \
               rect_list_contains(cliprects, ul_2) && rect_list_contains(cliprects, mainrect);
    if (!all_pass)
    {
        printf("Error in test_disjoint_corner()i: upper left\n");
    }

    // lower left
    cliprects = disjoint_union(mainrect, ll);
    rect_type ll_1(35, 35, 10, 5);
    rect_type ll_2(35, 40, 5, 5);
    all_pass = (cliprects.size() == 3) && rect_list_contains(cliprects, ll_1) && \
               rect_list_contains(cliprects, ll_2) && rect_list_contains(cliprects, mainrect);
    if (!all_pass)
    {
        printf("Error in test_disjoint_corner()i: upper left\n");
    }

    // upper right
    cliprects = disjoint_union(mainrect, ur);
    rect_type ur_1(55, 60, 10, 5);
    rect_type ur_2(60, 55, 5, 5);
    all_pass = (cliprects.size() == 3) && rect_list_contains(cliprects, ur_1) && \
               rect_list_contains(cliprects, ur_2) && rect_list_contains(cliprects, mainrect);
    if (!all_pass)
    {
        printf("Error in test_disjoint_corner()i: upper right\n");
    }

    // lower right
    cliprects = disjoint_union(mainrect, lr);
    rect_type lr_1(55, 35, 10, 5);
    rect_type lr_2(60, 40, 5, 5);
    all_pass = (cliprects.size() == 3) && rect_list_contains(cliprects, lr_1) && \
               rect_list_contains(cliprects, lr_2) && rect_list_contains(cliprects, mainrect);
    if (!all_pass)
    {
        printf("Error in test_disjoint_corner()i: lower right\n");
    }
}

// cases 8-11
void test_disjoint_edge_overlap()
{
    return;
    //rect_list_type cliprects;
    //rect_type mainrect(40,40,20,20);
}

void test_disjoint_union()
{
    test_disjoint_outside();
    test_disjoint_2_3();
    test_disjoint_corner();
}

}  // end namespace kiva
