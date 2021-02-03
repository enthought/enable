#ifndef KIVA_GL_EXCEPTIONS_H
#define KIVA_GL_EXCEPTIONS_H

namespace kiva_gl
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
