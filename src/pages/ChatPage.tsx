import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertTriangle,
  ArrowUp,
  Check,
  Database,
  Eraser,
  FileText,
  Filter,
  Loader2,
  Menu,
  MessageSquare,
  Pencil,
  RotateCcw,
  Square,
  Upload,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Link, useNavigate, useOutletContext, useParams, useSearchParams } from 'react-router-dom';
import { toast } from 'sonner';
import { ChatMessageBubble } from '../components/ChatMessageBubble';
import { Button } from '../components/ui/Button';
import { cn, getTimeGreeting } from '../lib/utils';
import { api, ApiClientError } from '../services/api';
import type { ChatMessage, ChatMode, Citation, PdfPreviewTarget } from '../types/api';

interface WorkspaceOutletContext {
  onOpenMobileNav: () => void;
  onOpenPdfPreview: (target: PdfPreviewTarget) => void;
}

const SUGGESTION_CARDS: {
  title: string;
  prompt: string;
  mode: ChatMode;
  category: string;
}[] = [
  {
    title: 'Summarize Q3 revenue & latency gains',
    prompt:
      'What was Meridian Cloud Q3 2026 revenue, ARR, and how did median vector retrieval latency change?',
    mode: 'documents',
    category: 'Document RAG',
  },
  {
    title: 'Verify AI governance & deletion SLA',
    prompt:
      'What are the data classification tiers and the document deletion SLA in the enterprise AI governance policy?',
    mode: 'documents',
    category: 'Policy Audit',
  },
  {
    title: 'Test hallucination boundary',
    prompt:
      'According to the uploaded documents, what is Meridian Cloud employee parental leave policy in Brazil?',
    mode: 'documents',
    category: 'Evidence Guard',
  },
  {
    title: 'Compare hybrid RAG retrieval architectures',
    prompt:
      'Compare dense vector search, BM25 lexical retrieval, and hybrid reranking for enterprise PDF question answering. Include a concise Python example.',
    mode: 'general',
    category: 'General AI',
  },
];

export function ChatPage() {
  const { id: routeConversationId } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { onOpenMobileNav, onOpenPdfPreview } = useOutletContext<WorkspaceOutletContext>();

  const [mode, setMode] = useState<ChatMode>(() =>
    searchParams.get('mode') === 'documents' ? 'documents' : 'general'
  );
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>(() => {
    const docParam = searchParams.get('doc');
    return docParam ? [docParam] : [];
  });
  const [showDocPicker, setShowDocPicker] = useState(false);

  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [streamingCitations, setStreamingCitations] = useState<Citation[]>([]);
  const [optimisticUserMsg, setOptimisticUserMsg] = useState<ChatMessage | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const [failedTurn, setFailedTurn] = useState<{
    prompt: string;
    mode: ChatMode;
    persisted: boolean;
  } | null>(null);

  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const docPickerRef = useRef<HTMLDivElement | null>(null);

  // Sync mode/doc query params when coming from Knowledge Base ("Chat with doc")
  useEffect(() => {
    const docParam = searchParams.get('doc');
    const modeParam = searchParams.get('mode');
    if (modeParam === 'documents') {
      setMode('documents');
    }
    if (docParam) {
      setMode('documents');
      setSelectedDocIds([docParam]);
      setSearchParams({}, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  // Close document picker popover on outside click
  useEffect(() => {
    if (!showDocPicker) return;
    const handleClickOutside = (event: MouseEvent) => {
      if (docPickerRef.current && !docPickerRef.current.contains(event.target as Node)) {
        setShowDocPicker(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showDocPicker]);

  const {
    data: documents = [],
    isFetched: isDocsFetched,
    isFetching: isDocsFetching,
  } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.listDocuments(),
  });

  const readyDocuments = useMemo(
    () => documents.filter((d) => d.status === 'ready'),
    [documents]
  );

  // Keep selectedDocIds clean only after the documents list has settled so newly uploaded ?doc= scopes are preserved
  useEffect(() => {
    if (!isDocsFetched || isDocsFetching) return;
    const validIds = new Set(readyDocuments.map((d) => d.id));
    setSelectedDocIds((prev) => {
      const next = prev.filter((id) => validIds.has(id));
      return next.length === prev.length ? prev : next;
    });
  }, [readyDocuments, isDocsFetched, isDocsFetching]);

  const {
    data: conversationData,
    isLoading: isLoadingConvo,
    error: convoError,
  } = useQuery({
    queryKey: ['conversation', routeConversationId],
    queryFn: () => api.getConversation(routeConversationId!),
    enabled: Boolean(routeConversationId),
    retry: false,
  });

  // Sync active mode from loaded conversation
  useEffect(() => {
    if (conversationData?.conversation) {
      setMode(conversationData.conversation.mode);
      setTitleDraft(conversationData.conversation.title);
    }
  }, [conversationData?.conversation?.id]);

  // Clear transient states when switching between existing conversations
  const prevRouteConvoRef = useRef<string | undefined>(routeConversationId);
  useEffect(() => {
    const switchedFromExisting = Boolean(prevRouteConvoRef.current);
    prevRouteConvoRef.current = routeConversationId;
    if (switchedFromExisting || !routeConversationId) {
      setChatError(null);
      setFailedTurn(null);
    }
    setOptimisticUserMsg(null);
    setStreamingContent('');
    setStreamingCitations([]);
    setIsEditingTitle(false);
  }, [routeConversationId]);

  const messages = useMemo(() => {
    const base = routeConversationId ? conversationData?.messages ?? [] : [];
    if (optimisticUserMsg && !base.some((m) => m.id === optimisticUserMsg.id)) {
      return [...base, optimisticUserMsg];
    }
    return base;
  }, [routeConversationId, conversationData?.messages, optimisticUserMsg]);

  // Smooth auto-scroll only during active sending/streaming or newly added turns
  const prevMsgCountRef = useRef(0);
  useEffect(() => {
    const grew = messages.length > prevMsgCountRef.current && prevMsgCountRef.current > 0;
    prevMsgCountRef.current = messages.length;
    if (isSending || streamingContent || grew) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages.length, streamingContent, isSending]);

  // Reset message count tracker when switching conversations
  useEffect(() => {
    prevMsgCountRef.current = 0;
  }, [routeConversationId]);

  // Auto-resize composer textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [input]);

  const handleStopStreaming = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsSending(false);
    setStreamingContent('');
    setStreamingCitations([]);
  };

  const sendMessage = async (
    promptText: string,
    overrideMode?: ChatMode,
    regenerate = false
  ) => {
    const trimmed = promptText.trim();
    if (!trimmed || isSending) return;

    const effectiveMode = overrideMode ?? mode;
    setChatError(null);
    setFailedTurn(null);
    setIsSending(true);
    setStreamingContent('');
    setStreamingCitations([]);

    if (!regenerate) {
      setInput('');
      setOptimisticUserMsg({
        id: `temp-user-${Date.now()}`,
        conversationId: routeConversationId || 'pending',
        role: 'user',
        content: trimmed,
        mode: effectiveMode,
        citations: [],
        insufficientEvidence: false,
        createdAt: new Date().toISOString(),
      });
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let resolvedConvoId = routeConversationId;

    try {
      await api.sendChatStream(
        {
          conversationId: routeConversationId,
          message: trimmed,
          mode: effectiveMode,
          documentIds:
            effectiveMode === 'documents' && selectedDocIds.length > 0
              ? selectedDocIds
              : undefined,
          regenerate,
        },
        {
          signal: controller.signal,
          onMeta: (meta) => {
            resolvedConvoId = meta.conversation.id;
            setOptimisticUserMsg(meta.userMessage);
            setStreamingCitations(meta.citations);
            queryClient.invalidateQueries({ queryKey: ['conversations'] });
          },
          onDelta: (delta) => {
            setStreamingContent((prev) => prev + delta);
          },
          onDone: (done) => {
            resolvedConvoId = done.conversation.id;
            setOptimisticUserMsg(null);
            setStreamingContent('');
            setStreamingCitations([]);
            queryClient.invalidateQueries({ queryKey: ['conversations'] });
            queryClient.invalidateQueries({ queryKey: ['conversation', resolvedConvoId] });
            if (!routeConversationId && resolvedConvoId) {
              navigate(`/chat/${resolvedConvoId}`, { replace: true });
            }
          },
        }
      );
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') {
        toast.info('Generation stopped.');
      } else {
        const message =
          err instanceof ApiClientError
            ? err.message
            : 'Unable to reach the DocuMind AI backend. Please check your connection and try again.';
        setChatError(message);
        setFailedTurn({
          prompt: trimmed,
          mode: effectiveMode,
          persisted: Boolean(resolvedConvoId),
        });
        toast.error(message);
        if (resolvedConvoId) {
          queryClient.invalidateQueries({ queryKey: ['conversation', resolvedConvoId] });
          if (!routeConversationId) {
            navigate(`/chat/${resolvedConvoId}`, { replace: true });
          }
        }
      }
      setOptimisticUserMsg(null);
      setStreamingContent('');
      setStreamingCitations([]);
    } finally {
      abortControllerRef.current = null;
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const handleRegenerate = () => {
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');
    if (!lastUser) return;
    sendMessage(lastUser.content, mode, true);
  };

  const handleClearChat = async () => {
    if (!routeConversationId || isSending) return;
    try {
      await api.clearConversation(routeConversationId);
      queryClient.invalidateQueries({ queryKey: ['conversation', routeConversationId] });
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      toast.success('Conversation cleared.');
    } catch {
      toast.error('Failed to clear conversation.');
    }
  };

  const handleSaveTitle = async () => {
    setIsEditingTitle(false);
    if (!routeConversationId) return;
    const trimmed = titleDraft.trim();
    if (!trimmed || trimmed === conversationData?.conversation.title) return;
    try {
      await api.updateConversation(routeConversationId, { title: trimmed });
      queryClient.invalidateQueries({ queryKey: ['conversation', routeConversationId] });
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    } catch {
      toast.error('Could not rename conversation.');
    }
  };

  const toggleDocSelection = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const lastAssistantId = useMemo(() => {
    const assistants = messages.filter((m) => m.role === 'assistant');
    return assistants.length > 0 ? assistants[assistants.length - 1].id : null;
  }, [messages]);

  const activeTitle = routeConversationId
    ? conversationData?.conversation.title ?? 'Conversation'
    : 'New Conversation';

  return (
    <div className="flex h-full min-h-0 flex-col bg-slate-50 dark:bg-[#0B0F19]">
      {/* Clean Top Workspace Bar */}
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-slate-200/80 bg-white/80 px-4 backdrop-blur-xs dark:border-slate-800/80 dark:bg-[#0B0F19]/90 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={onOpenMobileNav}
            aria-label="Open navigation menu"
            className="h-8 w-8 md:hidden"
          >
            <Menu className="h-4 w-4" />
          </Button>

          {routeConversationId && isEditingTitle ? (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSaveTitle();
              }}
              className="flex items-center gap-1.5"
            >
              <input
                type="text"
                value={titleDraft}
                onChange={(e) => setTitleDraft(e.target.value)}
                onBlur={handleSaveTitle}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') setIsEditingTitle(false);
                }}
                autoFocus
                aria-label="Edit conversation title"
                className="h-8 w-56 rounded-lg border border-indigo-400 bg-white px-2.5 text-sm font-medium text-slate-900 focus:outline-none dark:border-indigo-500 dark:bg-slate-900 dark:text-white sm:w-72"
              />
              <Button type="submit" variant="ghost" size="icon" className="h-8 w-8" aria-label="Save title">
                <Check className="h-4 w-4 text-indigo-500" />
              </Button>
            </form>
          ) : (
            <div className="flex min-w-0 items-center gap-2">
              <h1 className="truncate font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                {activeTitle}
              </h1>
              {routeConversationId && (
                <button
                  type="button"
                  onClick={() => {
                    setTitleDraft(activeTitle);
                    setIsEditingTitle(true);
                  }}
                  aria-label="Edit conversation title"
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200 cursor-pointer"
                >
                  <Pencil className="h-3 w-3" />
                </button>
              )}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {routeConversationId && messages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearChat}
              disabled={isSending}
              aria-label="Clear chat messages"
            >
              <Eraser className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Clear Chat</span>
            </Button>
          )}
          <Link
            to="/knowledge-base"
            className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50 dark:border-slate-800 dark:bg-[#111827] dark:text-slate-200 dark:hover:bg-slate-800 whitespace-nowrap"
          >
            <Upload className="h-3.5 w-3.5 text-indigo-500 dark:text-indigo-400" />
            <span>Upload PDF</span>
          </Link>
        </div>
      </header>

      {/* Main Message Scroll Viewport */}
      <main className="min-h-0 flex-1 overflow-y-auto">
        {routeConversationId && isLoadingConvo ? (
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-10 sm:px-6">
            <div className="ml-auto h-14 w-2/3 animate-pulse rounded-2xl bg-slate-200/70 dark:bg-slate-800/70" />
            <div className="space-y-2.5">
              <div className="h-4 w-1/3 animate-pulse rounded bg-slate-200/70 dark:bg-slate-800/70" />
              <div className="h-4 w-full animate-pulse rounded bg-slate-200/70 dark:bg-slate-800/70" />
              <div className="h-4 w-5/6 animate-pulse rounded bg-slate-200/70 dark:bg-slate-800/70" />
            </div>
          </div>
        ) : routeConversationId && convoError ? (
          <div className="mx-auto flex max-w-md flex-col items-center justify-center px-6 py-24 text-center">
            <AlertTriangle className="mb-3 h-7 w-7 text-amber-500" />
            <h2 className="font-display text-base font-semibold text-slate-900 dark:text-slate-100">
              Conversation not found
            </h2>
            <p className="mt-1.5 text-sm text-slate-500 dark:text-slate-400">
              This conversation may have been deleted or the URL is invalid.
            </p>
            <Button className="mt-5" onClick={() => navigate('/')}>
              Start New Conversation
            </Button>
          </div>
        ) : messages.length === 0 && !isSending ? (
          /* Welcome Empty State */
          <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center px-4 py-10 sm:px-6">
            <div className="mb-8">
              <p className="mb-2 text-xs font-medium text-indigo-600 dark:text-indigo-400">
                DocuMind AI · Your Knowledge. Intelligent Answers.
              </p>
              <h2 className="font-display text-2xl font-semibold tracking-tight text-slate-950 dark:text-slate-50 sm:text-3xl">
                {getTimeGreeting()}. What would you like to analyze?
              </h2>
              <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600 dark:text-slate-400">
                Ask open engineering questions in General AI mode, or switch to Chat With
                Documents to query your indexed PDFs with verifiable page-level citations.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {SUGGESTION_CARDS.map((card) => (
                <button
                  key={card.title}
                  type="button"
                  onClick={() => {
                    setMode(card.mode);
                    sendMessage(card.prompt, card.mode);
                  }}
                  className="group flex flex-col items-start gap-2 rounded-xl border border-slate-200 bg-white p-4 text-left transition-all duration-150 hover:border-indigo-300 hover:bg-indigo-50/30 dark:border-slate-800 dark:bg-[#111827] dark:hover:border-indigo-500/40 dark:hover:bg-[#151E32] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 cursor-pointer"
                >
                  <div className="flex w-full items-center justify-between gap-2">
                    <span className="text-xs font-medium text-indigo-600 dark:text-indigo-400">
                      {card.category}
                    </span>
                    <ArrowUp className="h-3.5 w-3.5 rotate-45 text-slate-300 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-indigo-500 dark:text-slate-600 dark:group-hover:text-indigo-400" />
                  </div>
                  <span className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                    {card.title}
                  </span>
                  <span className="line-clamp-2 text-xs leading-5 text-slate-500 dark:text-slate-400">
                    {card.prompt}
                  </span>
                </button>
              ))}
            </div>

            {readyDocuments.length > 0 && (
              <div className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
                <span className="font-medium text-slate-700 dark:text-slate-300">
                  Indexed Knowledge Base:
                </span>
                <span className="tabular-nums">
                  {readyDocuments.length}{' '}
                  {readyDocuments.length === 1 ? 'PDF document' : 'PDF documents'} ready
                </span>
                <span aria-hidden="true">·</span>
                <button
                  type="button"
                  onClick={() => setMode('documents')}
                  className="font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 cursor-pointer"
                >
                  Switch to Document Mode
                </button>
              </div>
            )}
          </div>
        ) : (
          /* Active Conversation Thread */
          <div className="mx-auto max-w-3xl space-y-7 px-4 py-7 sm:px-6">
            {messages.map((msg) => (
              <ChatMessageBubble
                key={msg.id}
                message={msg}
                isLastAssistant={msg.id === lastAssistantId && !isSending}
                onRegenerate={handleRegenerate}
                onSelectCitation={onOpenPdfPreview}
                disabledActions={isSending}
              />
            ))}

            {/* Active Streaming / Thinking Assistant Bubble */}
            {isSending && (
              streamingContent ? (
                <ChatMessageBubble
                  message={{
                    id: 'streaming-assistant',
                    conversationId: routeConversationId || 'pending',
                    role: 'assistant',
                    content: streamingContent,
                    mode,
                    citations: streamingCitations,
                    insufficientEvidence: false,
                    createdAt: new Date().toISOString(),
                  }}
                  isStreaming
                  onSelectCitation={onOpenPdfPreview}
                />
              ) : (
                <div className="flex items-center gap-3 text-sm text-slate-500 dark:text-slate-400">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#131B2E]">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-500" />
                  </div>
                  <span>
                    {mode === 'documents'
                      ? 'Searching indexed PDF passages and grounding answer…'
                      : 'Thinking…'}
                  </span>
                </div>
              )
            )}

            {/* Friendly Error Banner (distinct from insufficient evidence) */}
            {chatError && (
              <div
                role="alert"
                className="flex items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50/80 px-4 py-3 text-sm text-red-800 dark:border-red-900/60 dark:bg-red-950/35 dark:text-red-200"
              >
                <div className="flex items-start gap-2.5">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" />
                  <span>{chatError}</span>
                </div>
                <div className="flex shrink-0 items-center gap-3">
                  {failedTurn && (
                    <button
                      type="button"
                      disabled={isSending}
                      onClick={() =>
                        sendMessage(failedTurn.prompt, failedTurn.mode, failedTurn.persisted)
                      }
                      className="inline-flex items-center gap-1 rounded-md border border-red-300/80 bg-white/80 px-2.5 py-1 text-xs font-medium text-red-700 transition-colors hover:bg-red-100/70 disabled:opacity-50 dark:border-red-800/70 dark:bg-red-950/60 dark:text-red-200 dark:hover:bg-red-900/50 cursor-pointer"
                    >
                      <RotateCcw className="h-3 w-3" />
                      <span>Retry</span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      setChatError(null);
                      setFailedTurn(null);
                    }}
                    className="shrink-0 text-xs font-medium underline underline-offset-2 hover:opacity-80 cursor-pointer"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* Bottom Message Composer Dock */}
      <div className="shrink-0 border-t border-slate-200/80 bg-white/90 px-4 py-3.5 backdrop-blur-xs dark:border-slate-800/80 dark:bg-[#0B0F19]/95 sm:px-6">
        <div className="mx-auto max-w-3xl">
          {/* Mode selector & document scope bar */}
          <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
            <div
              role="tablist"
              aria-label="Chat mode"
              className="inline-flex items-center rounded-lg bg-slate-100 p-0.5 dark:bg-slate-900 border border-slate-200/70 dark:border-slate-800"
            >
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'general'}
                onClick={() => setMode('general')}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-colors cursor-pointer',
                  mode === 'general'
                    ? 'bg-white text-slate-950 shadow-2xs dark:bg-slate-800 dark:text-white'
                    : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                )}
              >
                <MessageSquare className="h-3.5 w-3.5" />
                <span>General AI</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'documents'}
                onClick={() => setMode('documents')}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-colors cursor-pointer',
                  mode === 'documents'
                    ? 'bg-indigo-600 text-white shadow-2xs dark:bg-indigo-500'
                    : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                )}
              >
                <Database className="h-3.5 w-3.5" />
                <span>Chat With Documents</span>
              </button>
            </div>

            {mode === 'documents' ? (
              <div className="relative" ref={docPickerRef}>
                <button
                  type="button"
                  onClick={() => setShowDocPicker((prev) => !prev)}
                  aria-expanded={showDocPicker}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50 dark:border-slate-800 dark:bg-[#111827] dark:text-slate-300 dark:hover:bg-slate-800 cursor-pointer whitespace-nowrap"
                >
                  <Filter className="h-3.5 w-3.5 text-indigo-500 dark:text-indigo-400" />
                  <span className="tabular-nums">
                    {readyDocuments.length === 0
                      ? 'No PDFs indexed'
                      : selectedDocIds.length === 0
                        ? `All ${readyDocuments.length} indexed PDFs`
                        : `${selectedDocIds.length} selected`}
                  </span>
                </button>

                {showDocPicker && (
                  <div className="absolute right-0 bottom-10 z-30 w-72 rounded-xl border border-slate-200 bg-white p-2 shadow-xl dark:border-slate-800 dark:bg-[#111827]">
                    <div className="flex items-center justify-between px-2 py-1.5 text-xs font-semibold text-slate-700 dark:text-slate-200">
                      <span>Retrieval Scope</span>
                      {selectedDocIds.length > 0 && (
                        <button
                          type="button"
                          onClick={() => setSelectedDocIds([])}
                          className="font-normal text-indigo-600 hover:underline dark:text-indigo-400 cursor-pointer"
                        >
                          Use all
                        </button>
                      )}
                    </div>
                    {readyDocuments.length === 0 ? (
                      <div className="px-2 py-3 text-xs text-slate-500 dark:text-slate-400">
                        Upload PDFs in Knowledge Base first.
                      </div>
                    ) : (
                      <div className="max-h-48 space-y-0.5 overflow-y-auto">
                        {readyDocuments.map((doc) => {
                          const checked = selectedDocIds.includes(doc.id);
                          return (
                            <label
                              key={doc.id}
                              className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800/70"
                            >
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => toggleDocSelection(doc.id)}
                                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                              />
                              <FileText className="h-3.5 w-3.5 shrink-0 text-slate-400" />
                              <span className="truncate flex-1">{doc.filename}</span>
                              <span className="font-mono text-[11px] text-slate-400 tabular-nums">
                                {doc.pageCount}p
                              </span>
                            </label>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <span className="text-xs text-slate-400 dark:text-slate-500">
                Open LLM conversation · Context-aware
              </span>
            )}
          </div>

          {/* Input container */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              sendMessage(input);
            }}
            className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-xs transition-colors focus-within:border-indigo-400 dark:border-slate-800 dark:bg-[#111827] dark:focus-within:border-indigo-500/70"
          >
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isSending}
              placeholder={
                mode === 'documents'
                  ? 'Ask a question grounded in your uploaded PDF documents…'
                  : 'Ask DocuMind AI anything…'
              }
              aria-label="Message input"
              className="max-h-44 min-h-[40px] flex-1 resize-none bg-transparent px-2.5 py-2 text-sm leading-6 text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:opacity-60 dark:text-slate-100 dark:placeholder:text-slate-500"
            />

            {isSending ? (
              <Button
                type="button"
                variant="secondary"
                size="icon"
                onClick={handleStopStreaming}
                aria-label="Stop generating"
                className="h-9 w-9 shrink-0 rounded-xl"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </Button>
            ) : (
              <Button
                type="submit"
                variant="primary"
                size="icon"
                disabled={!input.trim()}
                aria-label="Send message"
                className="h-9 w-9 shrink-0 rounded-xl"
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
            )}
          </form>

          <p className="mt-1.5 text-center text-[11px] text-slate-400 dark:text-slate-500">
            {mode === 'documents'
              ? 'Document mode answers strictly from retrieved PDF evidence with page citations. Press Enter to send, Shift+Enter for a new line.'
              : 'Press Enter to send, Shift+Enter for a new line.'}
          </p>
        </div>
      </div>
    </div>
  );
}
