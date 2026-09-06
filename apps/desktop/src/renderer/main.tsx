import { Component, StrictMode } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./styles.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
    throw new Error("Root container not found");
}

class RendererErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
    public state: { error: Error | null } = { error: null };

    public static getDerivedStateFromError(error: Error) {
        return { error };
    }

    public componentDidCatch(error: Error, info: ErrorInfo): void {
        console.error("Renderer crashed", error, info);
    }

    public render() {
        if (this.state.error) {
            return (
                <main className="renderer-error">
                    <h1>Renderer error</h1>
                    <p>App tidak blank lagi. Restart window atau kirim pesan error ini kalau masih muncul.</p>
                    <pre>{this.state.error.message}</pre>
                </main>
            );
        }

        return this.props.children;
    }
}

createRoot(rootElement).render(
    <StrictMode>
        <RendererErrorBoundary>
            <App />
        </RendererErrorBoundary>
    </StrictMode>
);
