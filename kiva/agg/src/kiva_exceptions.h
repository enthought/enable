// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_EXCEPTIONS_H
#define KIVA_EXCEPTIONS_H

namespace kiva
{
    // exception codes used in graphics_context
	enum {
		not_implemented_error = 0,
		ctm_rotation_error,
		bad_clip_state_error,
		even_odd_clip_error,
		clipping_path_unsupported
	};
}

#endif
