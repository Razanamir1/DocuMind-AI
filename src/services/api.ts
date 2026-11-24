import type {
  ChatMessage,
  ChatMode,
  Citation,
  ConversationItem,
  DocumentItem,
  HealthResponse,
  ProviderInfo,
  WorkspaceSettings,
} from '../types/api';

export class ApiClientError extends Error {
  code: string;
  status: number;

  constructor(message: string, code = 'API_ERROR', status = 500) {
    super(message);
    this.name = 'ApiClientError';
    this.code = code;
    this.status = status;
  }
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function parseJsonResponse<T>(res: Response): Promise<T> {
  const contentType = res.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await res.json() : null;

  if (!res.ok) {
    const errObj = data?.error;
    const detail = typeof data?.detail === 'string' ? data.detail : undefined;
    throw new ApiClientError(
      errObj?.message || detail || `Request failed with status ${res.status}`,
      errObj?.code || 'HTTP_ERROR',
      res.status
    );
  }
  return data as T;
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(apiUrl('/api/v1/health'));
    return parseJsonResponse<HealthResponse>(res);
  },

  async getSettings(): Promise<{ settings: WorkspaceSettings; provider: ProviderInfo }> {
    const res = await fetch(apiUrl('/api/v1/settings'));
    return parseJsonResponse<{ settings: WorkspaceSettings; provider: ProviderInfo }>(res);
  },

  async updateSettings(patch: Partial<WorkspaceSettings>): Promise<{ settings: WorkspaceSettings }> {
    const res = await fetch(apiUrl('/api/v1/settings'), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    return parseJsonResponse<{ settings: WorkspaceSettings }>(res);
  },

  async listConversations(): Promise<ConversationItem[]> {
    const res = await fetch(apiUrl('/api/v1/conversations'));
    const data = await parseJsonResponse<{ conversations: ConversationItem[] }>(res);
    return data.conversations;
  },

  async createConversation(params: {
    title?: string;
    mode?: ChatMode;
  }): Promise<ConversationItem> {
    const res = await fetch(apiUrl('/api/v1/conversations'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await parseJsonResponse<{ conversation: ConversationItem }>(res);
    return data.conversation;
  },

  async getConversation(
    id: string
  ): Promise<{ conversation: ConversationItem; messages: ChatMessage[] }> {
    const res = await fetch(apiUrl(`/api/v1/conversations/${encodeURIComponent(id)}`));
    return parseJsonResponse<{ conversation: ConversationItem; messages: ChatMessage[] }>(res);
  },

  async updateConversation(
    id: string,
    patch: { title?: string; mode?: ChatMode }
  ): Promise<ConversationItem> {
    const res = await fetch(apiUrl(`/api/v1/conversations/${encodeURIComponent(id)}`), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    const data = await parseJsonResponse<{ conversation: ConversationItem }>(res);
    return data.conversation;
  },

  async deleteConversation(id: string): Promise<void> {
    const res = await fetch(apiUrl(`/api/v1/conversations/${encodeURIComponent(id)}`), {
      method: 'DELETE',
    });
    await parseJsonResponse(res);
  },

  async clearConversation(id: string): Promise<void> {
    const res = await fetch(apiUrl(`/api/v1/conversations/${encodeURIComponent(id)}/clear`), {
      method: 'POST',
    });
    await parseJsonResponse(res);
  },

  async listDocuments(query?: string, status?: string): Promise<DocumentItem[]> {
    const params = new URLSearchParams();
    if (query?.trim()) params.set('q', query.trim());
    if (status && status !== 'all') params.set('status', status);
    const qs = params.toString();
    const res = await fetch(apiUrl(`/api/v1/documents${qs ? `?${qs}` : ''}`));
    const data = await parseJsonResponse<{ documents: DocumentItem[] }>(res);
    return data.documents;
  },

  async uploadDocument(
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<DocumentItem> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append('file', file);

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const pct = Math.min(92, Math.round((event.loaded / event.total) * 90));
          onProgress(pct);
        }
      });

      xhr.addEventListener('load', () => {
        try {
          const data = JSON.parse(xhr.responseText || '{}');
          if (xhr.status >= 200 && xhr.status < 300 && data.document) {
            onProgress?.(100);
            resolve(data.document as DocumentItem);
          } else {
            reject(
              new ApiClientError(
                data?.error?.message || 'Document upload failed.',
                data?.error?.code || 'UPLOAD_FAILED',
                xhr.status
              )
            );
          }
        } catch {
          reject(new ApiClientError('Invalid server response during upload.', 'PARSE_ERROR', xhr.status));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new ApiClientError('Network error while uploading PDF.', 'NETWORK_ERROR', 0));
      });

      xhr.open('POST', apiUrl('/api/v1/documents/upload'));
      xhr.send(formData);
    });
  },

  async deleteDocument(id: string): Promise<void> {
    const res = await fetch(apiUrl(`/api/v1/documents/${encodeURIComponent(id)}`), {
      method: 'DELETE',
    });
    await parseJsonResponse(res);
  },

  async getDocumentPage(
    id: string,
    pageNumber: number
  ): Promise<{ document: DocumentItem; page: { documentId: string; pageNumber: number; content: string } }> {
    const res = await fetch(
      apiUrl(`/api/v1/documents/${encodeURIComponent(id)}/pages/${pageNumber}`)
    );
    return parseJsonResponse(res);
  },

  getDocumentFileUrl(id: string): string {
    return apiUrl(`/api/v1/documents/${encodeURIComponent(id)}/file`);
  },

  async sendChatStream(
    payload: {
      conversationId?: string;
      message: string;
      mode: ChatMode;
      documentIds?: string[];
      regenerate?: boolean;
    },
    handlers: {
      onMeta?: (meta: {
        conversation: ConversationItem;
        userMessage: ChatMessage;
        mode: ChatMode;
        citations: Citation[];
      }) => void;
      onDelta?: (text: string) => void;
      onDone?: (result: {
        conversation: ConversationItem;
        assistantMessage: ChatMessage;
      }) => void;
      signal?: AbortSignal;
    }
  ): Promise<void> {
    const res = await fetch(apiUrl('/api/v1/chat'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ ...payload, stream: true }),
      signal: handlers.signal,
    });

    if (!res.ok) {
      await parseJsonResponse(res);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new ApiClientError('Streaming response body unavailable.', 'STREAM_UNAVAILABLE', 500);
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex = buffer.indexOf('\n\n');
      while (separatorIndex !== -1) {
        const rawBlock = buffer.slice(0, separatorIndex).trim();
        buffer = buffer.slice(separatorIndex + 2);

        if (rawBlock) {
          let eventName = 'message';
          const dataLines: string[] = [];

          for (const line of rawBlock.split('\n')) {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5).trim());
            }
          }

          if (dataLines.length > 0) {
            const parsed = JSON.parse(dataLines.join('\n'));
            if (eventName === 'meta') {
              handlers.onMeta?.(parsed);
            } else if (eventName === 'delta') {
              handlers.onDelta?.(parsed.text || '');
            } else if (eventName === 'done') {
              handlers.onDone?.(parsed);
            } else if (eventName === 'error') {
              throw new ApiClientError(
                parsed.message || 'AI response stream failed.',
                parsed.code || 'STREAM_ERROR',
                502
              );
            }
          }
        }

        separatorIndex = buffer.indexOf('\n\n');
      }
    }
  },
};
