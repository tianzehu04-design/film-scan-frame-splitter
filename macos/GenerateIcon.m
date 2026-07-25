#import <Cocoa/Cocoa.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) return 1;

        NSImage *image = [[NSImage alloc] initWithSize:NSMakeSize(1024, 1024)];
        [image lockFocus];

        NSBezierPath *background = [NSBezierPath
            bezierPathWithRoundedRect:NSMakeRect(48, 48, 928, 928)
            xRadius:220
            yRadius:220
        ];
        [[NSColor colorWithCalibratedRed:0.29 green:0.20 blue:0.16 alpha:1] setFill];
        [background fill];

        NSBezierPath *body = [NSBezierPath
            bezierPathWithRoundedRect:NSMakeRect(185, 250, 654, 480)
            xRadius:115
            yRadius:115
        ];
        [[NSColor colorWithCalibratedRed:0.96 green:0.93 blue:0.87 alpha:1] setFill];
        [body fill];

        NSBezierPath *top = [NSBezierPath bezierPath];
        [top moveToPoint:NSMakePoint(310, 710)];
        [top lineToPoint:NSMakePoint(375, 820)];
        [top lineToPoint:NSMakePoint(610, 820)];
        [top lineToPoint:NSMakePoint(675, 710)];
        [top closePath];
        [top fill];

        NSBezierPath *lensOuter = [NSBezierPath
            bezierPathWithOvalInRect:NSMakeRect(327, 306, 370, 370)
        ];
        [[NSColor colorWithCalibratedRed:0.29 green:0.20 blue:0.16 alpha:1] setFill];
        [lensOuter fill];

        NSBezierPath *lensInner = [NSBezierPath
            bezierPathWithOvalInRect:NSMakeRect(392, 371, 240, 240)
        ];
        [[NSColor colorWithCalibratedRed:0.60 green:0.32 blue:0.21 alpha:1] setFill];
        [lensInner fill];

        NSBezierPath *highlight = [NSBezierPath
            bezierPathWithOvalInRect:NSMakeRect(430, 525, 70, 70)
        ];
        [[NSColor colorWithCalibratedRed:0.96 green:0.93 blue:0.87 alpha:0.95] setFill];
        [highlight fill];

        NSBezierPath *indicator = [NSBezierPath
            bezierPathWithOvalInRect:NSMakeRect(720, 620, 58, 58)
        ];
        [[NSColor colorWithCalibratedRed:0.76 green:0.54 blue:0.36 alpha:1] setFill];
        [indicator fill];

        [image unlockFocus];

        NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithData:image.TIFFRepresentation];
        NSData *png = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        return [png writeToFile:path atomically:YES] ? 0 : 1;
    }
}
