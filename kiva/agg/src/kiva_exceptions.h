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
