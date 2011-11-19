#import <Cocoa/Cocoa.h>

void *
get_cg_context_ref(void *view_ptr)
{
    NSView *view = (NSView *)view_ptr;
    return [[[view window] graphicsContext] graphicsPort];
}
