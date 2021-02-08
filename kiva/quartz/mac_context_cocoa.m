// (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#import <Cocoa/Cocoa.h>

void *
get_cg_context_ref(void *view_ptr)
{
    id object = (id)view_ptr;

    if ([object isKindOfClass:[NSView class]])
    {
        NSView *view = (NSView *)object;
        return [[[view window] graphicsContext] graphicsPort];
    }
    return (void *)0;
}
