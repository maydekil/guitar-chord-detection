export interface VocalRemovalSuccess {
    version: string;
    source: {
        path: string;
    };
    audio: {
        path: string;
        streamUrl?: string;
        stem: "instrumental";
    };
}

export interface VocalRemovalError {
    version: string;
    error: {
        code: string;
        message: string;
    };
}

export type VocalRemovalResult = VocalRemovalSuccess | VocalRemovalError;
