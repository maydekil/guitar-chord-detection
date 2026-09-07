import { fireEvent, screen } from "@testing-library/react";

export function setMediaTiming(audio: HTMLAudioElement, values: { duration: number; currentTime: number }): void {
    const { duration, currentTime } = values;
    Object.defineProperty(audio, "duration", { value: duration, configurable: true });
    Object.defineProperty(audio, "currentTime", { value: currentTime, writable: true, configurable: true });
}

export function activeChords(): string[] {
    return screen
        .getAllByTestId("timeline-segment")
        .filter((segment) => segment.getAttribute("data-active") === "true")
        .map((segment) => segment.getAttribute("data-chord") ?? "");
}

export async function openAnalyzerForm(): Promise<void> {
    fireEvent.click(await screen.findByRole("button", { name: "Add Song" }));
}
