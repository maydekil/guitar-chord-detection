import type { GenreEvaluationMetrics, GenreEvaluationResult, GenreEvaluationSuccess } from "@gcd/shared/genreEvaluation";

type GenreEvaluationPanelProps = {
    manifestFileName: string;
    manifestPath: string | null;
    result: GenreEvaluationResult | null;
    isEvaluating: boolean;
    status: string;
    onSelectManifest: () => void;
    onRunEvaluation: () => void;
};

export function GenreEvaluationPanel(props: GenreEvaluationPanelProps) {
    const success = props.result && !("error" in props.result) ? props.result : null;
    return (
        <section className="shell-card detector-panel genre-evaluation-panel" aria-label="genre evaluation">
            <div className="detector-topbar">
                <div className="detector-actions">
                    <button type="button" className="open-btn" onClick={props.onSelectManifest} disabled={props.isEvaluating}>
                        Open Manifest
                    </button>
                    <button type="button" className="open-btn" onClick={props.onRunEvaluation} disabled={!props.manifestPath || props.isEvaluating}>
                        {props.isEvaluating ? "Evaluating..." : "Run Evaluation"}
                    </button>
                </div>
                <p className="state-label">{success ? `Status: ${success.status}` : "Status: idle"}</p>
            </div>
            <div className="detector-summary genre-evaluation-summary">
                <div>
                    <p className="file-name">{props.manifestFileName}</p>
                    <p className="analysis-label" aria-live="polite">{props.status}</p>
                </div>
                {success ? (
                    <p className="active-chord">
                        Accuracy <span>{formatPercent(success.metrics.timeWeightedChordAccuracy)}</span>
                    </p>
                ) : null}
            </div>
            {props.result && "error" in props.result ? (
                <div className="genre-evaluation-error">
                    <strong>{props.result.error.code}</strong>
                    <span>{props.result.error.message}</span>
                </div>
            ) : null}
            {success ? <EvaluationReport result={success} /> : <EmptyReport />}
        </section>
    );
}

function EmptyReport() {
    return (
        <div className="detector-empty genre-evaluation-empty">
            <h2>Genre Evaluation</h2>
            <p>Pilih manifest JSON untuk menjalankan evaluasi ground-truth lintas genre.</p>
        </div>
    );
}

function EvaluationReport({ result }: { result: GenreEvaluationSuccess }) {
    const genreRows = Object.entries(result.genres).sort(([left], [right]) => left.localeCompare(right));
    return (
        <div className="genre-evaluation-report">
            <div className="genre-evaluation-kpis" aria-label="Corpus evaluation summary">
                <Metric label="Items" value={`${result.evaluatedItemCount}/${result.itemCount}`} />
                <Metric label="Failed" value={String(result.failedItemCount)} />
                <Metric label="Root" value={formatPercent(result.metrics.rootAccuracy)} />
                <Metric label="Quality" value={formatPercent(result.metrics.qualityAccuracy)} />
                <Metric label="False" value={String(result.metrics.falseTransitionCount)} />
                <Metric label="Missed" value={String(result.metrics.missedTransitionCount)} />
            </div>
            <section className="genre-evaluation-section" aria-label="Genre metrics">
                <h3>Genre Metrics</h3>
                <div className="genre-evaluation-table" role="table">
                    <div role="row" className="genre-evaluation-row header-row">
                        <span>Genre</span>
                        <span>Items</span>
                        <span>Chord</span>
                        <span>Root</span>
                        <span>Quality</span>
                        <span>False</span>
                        <span>Missed</span>
                    </div>
                    {genreRows.map(([genre, metrics]) => (
                        <div role="row" className="genre-evaluation-row" key={genre}>
                            <span>{genre}</span>
                            <span>{metrics.itemCount}</span>
                            <span>{formatPercent(metrics.timeWeightedChordAccuracy)}</span>
                            <span>{formatPercent(metrics.rootAccuracy)}</span>
                            <span>{formatPercent(metrics.qualityAccuracy)}</span>
                            <span>{metrics.falseTransitionCount}</span>
                            <span>{metrics.missedTransitionCount}</span>
                        </div>
                    ))}
                </div>
            </section>
            <section className="genre-evaluation-section" aria-label="Evaluation items">
                <h3>Items</h3>
                <div className="genre-evaluation-items">
                    {result.items.map((item) => (
                        <div key={`${item.genre}-${item.id}`} className="genre-evaluation-item" data-status={item.status}>
                            <div>
                                <strong>{item.id}</strong>
                                <small>{item.genre}</small>
                                {item.status === "pass" && item.metrics.confusionPairs?.length ? (
                                    <small className="genre-evaluation-confusions">
                                        {item.metrics.confusionPairs.slice(0, 3).map((pair) => pair.pair).join(" · ")}
                                    </small>
                                ) : null}
                            </div>
                            {item.status === "pass" ? (
                                <span>{formatPercent(item.metrics.timeWeightedChordAccuracy)}</span>
                            ) : (
                                <span>{item.error.code}</span>
                            )}
                        </div>
                    ))}
                </div>
            </section>
        </div>
    );
}

function Metric({ label, value }: { label: string; value: string }) {
    return (
        <div className="genre-evaluation-metric">
            <small>{label}</small>
            <strong>{value}</strong>
        </div>
    );
}

function formatPercent(value: GenreEvaluationMetrics[keyof GenreEvaluationMetrics]): string {
    return typeof value === "number" ? `${value.toFixed(1)}%` : "-";
}
