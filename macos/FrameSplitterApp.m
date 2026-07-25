#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>

@interface AppDelegate : NSObject <NSApplicationDelegate, WKUIDelegate, WKNavigationDelegate>
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) WKWebView *webView;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [self configureMainMenu];

    WKWebViewConfiguration *configuration = [[WKWebViewConfiguration alloc] init];
    configuration.websiteDataStore = [WKWebsiteDataStore defaultDataStore];
    [configuration.preferences setValue:@YES forKey:@"allowFileAccessFromFileURLs"];

    self.webView = [[WKWebView alloc] initWithFrame:NSZeroRect configuration:configuration];
    self.webView.UIDelegate = self;
    self.webView.navigationDelegate = self;
    [self.webView setValue:@NO forKey:@"drawsBackground"];

    self.window = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, 1440, 900)
        styleMask:NSWindowStyleMaskTitled |
                  NSWindowStyleMaskClosable |
                  NSWindowStyleMaskMiniaturizable |
                  NSWindowStyleMaskResizable |
                  NSWindowStyleMaskFullSizeContentView
        backing:NSBackingStoreBuffered
        defer:NO
    ];
    self.window.title = @"Frame Splitter";
    self.window.minSize = NSMakeSize(960, 640);
    self.window.contentView = self.webView;
    [self.window center];
    [self.window makeKeyAndOrderFront:nil];

    NSURL *resourcesURL = NSBundle.mainBundle.resourceURL;
    NSURL *indexURL = [NSBundle.mainBundle URLForResource:@"index"
                                            withExtension:@"html"
                                             subdirectory:@"WebApp"];
    if (!resourcesURL || !indexURL) {
        [self showLoadError:@"The application resources could not be found."];
        return;
    }

    [self.webView loadFileURL:indexURL allowingReadAccessToURL:resourcesURL];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (void)webView:(WKWebView *)webView
runOpenPanelWithParameters:(WKOpenPanelParameters *)parameters
initiatedByFrame:(WKFrameInfo *)frame
completionHandler:(void (^)(NSArray<NSURL *> * _Nullable URLs))completionHandler {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = parameters.allowsMultipleSelection;
    panel.allowedFileTypes = @[@"png", @"jpg", @"jpeg", @"tif", @"tiff"];
    [panel beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse response) {
        completionHandler(response == NSModalResponseOK ? panel.URLs : nil);
    }];
}

- (void)webView:(WKWebView *)webView
didFailNavigation:(WKNavigation *)navigation
      withError:(NSError *)error {
    [self showLoadError:error.localizedDescription];
}

- (void)webView:(WKWebView *)webView
didFailProvisionalNavigation:(WKNavigation *)navigation
      withError:(NSError *)error {
    [self showLoadError:error.localizedDescription];
}

- (void)showLoadError:(NSString *)message {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Frame Splitter could not start";
    alert.informativeText = message;
    alert.alertStyle = NSAlertStyleCritical;
    [alert runModal];
}

- (void)configureMainMenu {
    NSMenu *mainMenu = [[NSMenu alloc] init];

    NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
    [mainMenu addItem:appMenuItem];
    NSMenu *appMenu = [[NSMenu alloc] init];
    appMenuItem.submenu = appMenu;
    [appMenu addItemWithTitle:@"About Frame Splitter"
                       action:@selector(orderFrontStandardAboutPanel:)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit Frame Splitter"
                       action:@selector(terminate:)
                keyEquivalent:@"q"];

    NSMenuItem *editMenuItem = [[NSMenuItem alloc] init];
    [mainMenu addItem:editMenuItem];
    NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
    editMenuItem.submenu = editMenu;
    [editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
    [editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"];
    [editMenu addItem:[NSMenuItem separatorItem]];
    [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];

    NSMenuItem *windowMenuItem = [[NSMenuItem alloc] init];
    [mainMenu addItem:windowMenuItem];
    NSMenu *windowMenu = [[NSMenu alloc] initWithTitle:@"Window"];
    windowMenuItem.submenu = windowMenu;
    [windowMenu addItemWithTitle:@"Minimize"
                         action:@selector(performMiniaturize:)
                  keyEquivalent:@"m"];
    [windowMenu addItemWithTitle:@"Zoom"
                         action:@selector(performZoom:)
                  keyEquivalent:@""];
    NSApp.windowsMenu = windowMenu;

    NSApp.mainMenu = mainMenu;
}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *application = NSApplication.sharedApplication;
        AppDelegate *delegate = [[AppDelegate alloc] init];
        application.delegate = delegate;
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
        [application activateIgnoringOtherApps:YES];
        [application run];
    }
    return 0;
}
