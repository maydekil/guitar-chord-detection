import "@testing-library/jest-dom/vitest";

Object.defineProperty(HTMLMediaElement.prototype, "play", {
    configurable: true,
    value: function play() {
        return Promise.resolve();
    }
});

Object.defineProperty(HTMLMediaElement.prototype, "pause", {
    configurable: true,
    value: function pause() { }
});
