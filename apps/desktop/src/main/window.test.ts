import { describe, expect, it } from "vitest";

import { buildMainWindowOptions } from "./window";

describe("buildMainWindowOptions", () => {
    it("enforces secure webPreferences defaults", () => {
        const options = buildMainWindowOptions("/tmp/preload.js");

        expect(options.webPreferences?.preload).toBe("/tmp/preload.js");
        expect(options.webPreferences?.contextIsolation).toBe(true);
        expect(options.webPreferences?.nodeIntegration).toBe(false);
        expect(options.webPreferences?.sandbox).toBe(true);
    });
});
