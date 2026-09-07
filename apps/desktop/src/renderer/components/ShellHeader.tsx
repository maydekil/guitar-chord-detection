import type { ApiHealthState } from "../appTypes.js";
import { formatApiHealthLabel } from "../lib/apiStatus.js";

type ShellHeaderProps = {
    version: string;
    apiHealthState: ApiHealthState;
    apiBaseUrl: string;
    draftApiBaseUrl: string;
    isApiSettingsOpen: boolean;
    apiSettingsStatus: string;
    isTestingApiConfig: boolean;
    isSavingApiConfig: boolean;
    onToggleApiSettings: () => void;
    onDraftApiBaseUrlChange: (value: string) => void;
    onTestApiConfig: () => void;
    onSaveApiConfig: () => void;
};

export function ShellHeader({
    version,
    apiHealthState,
    apiBaseUrl,
    draftApiBaseUrl,
    isApiSettingsOpen,
    apiSettingsStatus,
    isTestingApiConfig,
    isSavingApiConfig,
    onToggleApiSettings,
    onDraftApiBaseUrlChange,
    onTestApiConfig,
    onSaveApiConfig,
}: ShellHeaderProps) {
    return (
        <header className="shell-header">
            <div className="shell-title-row">
                <h1>Guitar Chord Detector</h1>
                <span className={`mode-badge mode-badge-${apiHealthState}`}>{formatApiHealthLabel(apiHealthState)}</span>
                <span className="api-settings-menu">
                    <button
                        type="button"
                        className="api-settings-toggle"
                        onClick={onToggleApiSettings}
                        aria-expanded={isApiSettingsOpen}
                    >
                        ⚙ API
                    </button>
                    {isApiSettingsOpen ? (
                        <section className="api-settings-panel" aria-label="API settings">
                            <label>
                                <span>API URL</span>
                                <input
                                    type="url"
                                    value={draftApiBaseUrl}
                                    onChange={(event) => onDraftApiBaseUrlChange(event.currentTarget.value)}
                                    placeholder="Kosongkan untuk Local Mode"
                                />
                            </label>
                            <div className="api-settings-actions">
                                <button type="button" onClick={onTestApiConfig} disabled={isTestingApiConfig || isSavingApiConfig}>
                                    {isTestingApiConfig ? "Testing..." : "Test"}
                                </button>
                                <button type="button" onClick={onSaveApiConfig} disabled={isTestingApiConfig || isSavingApiConfig}>
                                    {isSavingApiConfig ? "Saving..." : "Save"}
                                </button>
                            </div>
                            <small>{apiSettingsStatus || (apiBaseUrl ? `Current: ${apiBaseUrl}` : "Current: Local Mode")}</small>
                        </section>
                    ) : null}
                </span>
            </div>
            <p className="shell-meta">Desktop Shell v{version}</p>
        </header>
    );
}
