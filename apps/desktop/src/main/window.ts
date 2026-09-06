import type { BrowserWindowConstructorOptions } from "electron";

export function buildMainWindowOptions(preloadPath: string, iconPath: string): BrowserWindowConstructorOptions {
    return {
        width: 1200,
        height: 760,
        minWidth: 900,
        minHeight: 560,
        title: "Guitar Chord Detector",
        icon: iconPath,
        backgroundColor: "#f4f1ea",
        webPreferences: {
            preload: preloadPath,
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: true,
            devTools: true
        }
    };
}
