export type ChatMode = 'general' | 'documents';

export type DocumentStatus = 'uploading' | 'processing' | 'ready' | 'failed';

export interface Citation {
  id: string;
  documentId: string;
  filename: string;
  pageNumber: number;
  excerpt: string;
  score: number;
  chunkIndex: number;
}

export interface DocumentItem {
  id: string;
  filename: string;
  storedFilename: string;
  sha256: string;
  fileSizeBytes: number;
  pageCount: number;
  chunkCount: number;
  status: DocumentStatus;
  embeddingStatus: 'vector' | 'lexical_fallback' | 'pending';
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ConversationItem {
  id: string;
  title: string;
  mode: ChatMode;
  createdAt: string;
  updatedAt: string;
  messageCount?: number;
  lastMessagePreview?: string;
}

export interface ChatMessage {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant';
  content: string;
  mode: ChatMode;
  citations: Citation[];
  insufficientEvidence: boolean;
  createdAt: string;
}

export interface WorkspaceSettings {
  topK: number;
  minRelevanceScore: number;
  chunkSize: number;
  chunkOverlap: number;
  streamResponses: boolean;
}

export interface ProviderInfo {
  name: string;
  chatModel: string;
  embeddingModel: string;
  configured: boolean;
  backendEngine?: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
  provider: ProviderInfo;
  stats: {
    documentsTotal: number;
    documentsReady: number;
    documentsIndexed?: number;
    documentsPending?: number;
    indexingState?: 'ready' | 'indexing' | 'pending' | 'degraded';
    storageMode?: 'ephemeral-local';
    lastIndexingError?: string | null;
    totalChunks: number;
    chromaVectorCount?: number;
    embeddingDimension?: number | null;
    conversationsTotal: number;
  };
}

export interface PdfPreviewTarget {
  documentId: string;
  filename: string;
  pageNumber: number;
  excerpt?: string;
}
