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
