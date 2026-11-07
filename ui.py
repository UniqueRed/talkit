"""Native macOS window hosting the HTML interface (WKWebView) + a JS <-> Python bridge.

All methods here must run on the main thread.
"""

from __future__ import annotations

import json
from pathlib import Path

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSMenu,
    NSMenuItem,
    NSModalResponseOK,
    NSSavePanel,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSMakeRect, NSMakeSize, NSObject, NSURL
from WebKit import WKWebView, WKWebViewConfiguration

HTML_PATH = Path(__file__).with_name("ui.html")
WKScriptMessageHandler = objc.protocolNamed("WKScriptMessageHandler")


class _Bridge(NSObject, protocols=[WKScriptMessageHandler]):
    def initWithHandler_(self, handler):
        self = objc.super(_Bridge, self).init()
        if self is not None:
            self._handler = handler
        return self

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        try:
            self._handler(json.loads(str(message.body())))
        except Exception as e:  # never let a UI message take the app down
            print("ui message error:", repr(e), flush=True)


class _WindowDelegate(NSObject):
    def initWithOwner_(self, owner):
        self = objc.super(_WindowDelegate, self).init()
        if self is not None:
            self._owner = owner
        return self

    def windowShouldClose_(self, sender):
        self._owner.hide()   # closing just hides; the app keeps running in the menu bar
        return False


class _MenuTarget(NSObject):
    def initWithQuit_(self, on_quit):
        self = objc.super(_MenuTarget, self).init()
        if self is not None:
            self._on_quit = on_quit
        return self

    def quit_(self, sender):
        self._on_quit()


class MainWindow:
    def __init__(self, on_message, on_quit):
        self._ready = False
        self._queue: list[str] = []

        config = WKWebViewConfiguration.alloc().init()
        self._bridge = _Bridge.alloc().initWithHandler_(self._on_message(on_message))
        config.userContentController().addScriptMessageHandler_name_(self._bridge, "flow")

        rect = NSMakeRect(0, 0, 980, 660)
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                 | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False)
        self.window.setTitle_("Flow")
        self.window.setMinSize_(NSMakeSize(720, 440))
        self.window.setReleasedWhenClosed_(False)
        self.window.setFrameAutosaveName_("FlowMainWindow")
        if not self.window.setFrameUsingName_("FlowMainWindow"):
            self.window.center()
        self._delegate = _WindowDelegate.alloc().initWithOwner_(self)
        self.window.setDelegate_(self._delegate)

        self.webview = WKWebView.alloc().initWithFrame_configuration_(rect, config)
        self.webview.setAutoresizingMask_(2 | 16)  # flexible width + height
        if self.webview.respondsToSelector_("setInspectable:"):
            self.webview.setInspectable_(True)     # right-click → Inspect Element for debugging
        self.window.setContentView_(self.webview)
        self.webview.loadHTMLString_baseURL_(HTML_PATH.read_text(), NSURL.fileURLWithPath_(str(HTML_PATH.parent)))

        self._menu_target = _MenuTarget.alloc().initWithQuit_(on_quit)
        self._install_main_menu()

    def _on_message(self, handler):
        def wrapped(msg: dict):
            if msg.get("cmd") == "ready":
                self._ready = True
                for js in self._queue:
                    self._eval(js)
                self._queue.clear()
            handler(msg)
        return wrapped

    # ---- Python -> JS -----------------------------------------------------

    def send(self, obj: dict):
        js = f"window.flow && flow.receive({json.dumps(obj)})"
        if self._ready:
            self._eval(js)
        elif obj.get("type") != "level":
            self._queue.append(js)

    def _eval(self, js: str):
        self.webview.evaluateJavaScript_completionHandler_(js, None)

    # ---- window management ------------------------------------------------

    def show(self):
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyRegular)   # dock icon + menu bar while open
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def hide(self):
        self.window.orderOut_(None)
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # back to menu-bar-only

    @property
    def visible(self) -> bool:
        return bool(self.window.isVisible())

    def save_file(self, default_name: str, content: str) -> str | None:
        """Shows a native Save dialog; returns the saved path, or None if cancelled."""
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        panel = NSSavePanel.savePanel()
        panel.setNameFieldStringValue_(default_name)
        panel.setCanCreateDirectories_(True)
        if panel.runModal() != NSModalResponseOK:
            return None
        path = str(panel.URL().path())
        Path(path).write_text(content)
        return path

    def _install_main_menu(self):
        """Standard app/Edit/Window menus so ⌘C, ⌘V, ⌘A, ⌘W, ⌘Q work in the window."""
        main = NSMenu.alloc().init()

        def submenu(title, items):
            top = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
            menu = NSMenu.alloc().initWithTitle_(title)
            for label, action, key, target in items:
                if label is None:
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label, action, key)
                if target is not None:
                    item.setTarget_(target)
                menu.addItem_(item)
            top.setSubmenu_(menu)
            main.addItem_(top)

        submenu("Flow", [
            ("Hide Flow", "hide:", "h", None),
            (None, None, None, None),
            ("Quit Flow", "quit:", "q", self._menu_target),
        ])
        submenu("Edit", [
            ("Undo", "undo:", "z", None),
            ("Redo", "redo:", "Z", None),
            (None, None, None, None),
            ("Cut", "cut:", "x", None),
            ("Copy", "copy:", "c", None),
            ("Paste", "paste:", "v", None),
            ("Select All", "selectAll:", "a", None),
        ])
        submenu("Window", [
            ("Close", "performClose:", "w", None),
            ("Minimize", "performMiniaturize:", "m", None),
        ])
        NSApplication.sharedApplication().setMainMenu_(main)
