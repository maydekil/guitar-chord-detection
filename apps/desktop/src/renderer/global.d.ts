import type { DesktopApi } from "../preload/index";

declare global {
    interface Window {
        gcd?: DesktopApi;
    }
}

export { };
