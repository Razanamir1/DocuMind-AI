import {
  AlertCircle,
  BookOpen,
  Check,
  Copy,
  FileText,
  RefreshCw,
} from 'lucide-react';
import { motion } from 'motion/react';
import { useState } from 'react';
import { cn, formatRelativeDate } from '../lib/utils';
import type { ChatMessage, Citation, PdfPreviewTarget } from '../types/api';
import { MarkdownRenderer } from './MarkdownRenderer';

interface ChatMessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
  isLastAssistant?: boolean;
  onRegenerate?: () => void;
  onSelectCitation?: (target: PdfPreviewTarget) => void;
  disabledActions?: boolean;
}

export function ChatMessageBubble({
  message,
  isStreaming = false,
  isLastAssistant = false,
  onRegenerate,
  onSelectCitation,
  disabledActions = false,
}: ChatMessageBubbleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Ignore clipboard failures
    }
  };

  if (message.role === 'user') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        className="flex justify-end"
      >
        <div className="max-w-[82%] sm:max-w-[74%]">
          <div className="rounded-2xl rounded-br-md bg-indigo-600 px-4 py-3 text-[15px] leading-6 text-white shadow-xs dark:bg-indigo-500/95">
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          </div>
          <div className="mt-1.5 flex items-center justify-end gap-2 pr-1 text-[11px] text-slate-400 dark:text-slate-500 tabular-nums">
            <span>{message.mode === 'documents' ? 'Documents' : 'General AI'}</span>
            <span aria-hidden="true">·</span>
            <span>{formatRelativeDate(message.createdAt)}</span>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
      className="group flex items-start gap-3.5"
    >
      {/* Assistant mark */}
      <div
        className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-white text-indigo-600 shadow-2xs dark:border-slate-800 dark:bg-[#131B2E] dark:text-indigo-400"
        aria-hidden="true"
      >
        <BookOpen className="h-3.5 w-3.5" />
      </div>

      <div className="min-w-0 flex-1 max-w-[74ch]">
        {/* Quiet header metadata */}
        <div className="mb-1.5 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          <span className="font-display font-semibold text-slate-900 dark:text-slate-100">
            DocuMind AI
          </span>
          <span aria-hidden="true">·</span>
          <span>
            {message.mode === 'documents' ? 'Document Grounded' : 'General AI'}
          </span>
          {!isStreaming && (
            <>
              <span aria-hidden="true">·</span>
              <span className="tabular-nums">{formatRelativeDate(message.createdAt)}</span>
            </>
          )}
        </div>

        {/* Insufficient evidence callout */}
        {message.insufficientEvidence && (
          <div className="mb-3 flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-3.5 py-2.5 text-xs leading-5 text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
            <span>
              Insufficient supporting evidence was found in the active Knowledge Base documents.
              No external claims or citations were fabricated.
            </span>
          </div>
        )}

        {/* Markdown prose body */}
        <div className="min-w-0">
          <MarkdownRenderer content={message.content} />
          {isStreaming && (
            <span
              className="ml-1 inline-block h-4 w-1.5 align-middle rounded-xs bg-indigo-500 animate-pulse"
              aria-label="Generating response"
            />
          )}
        </div>

        {/* Verified Source Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-4 border-t border-slate-200/80 pt-3.5 dark:border-slate-800/80">
            <div className="mb-2 flex items-center justify-between">
              <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Sources
              </h4>
              <span className="text-[11px] text-slate-400 dark:text-slate-500 tabular-nums">
                {message.citations.length}{' '}
                {message.citations.length === 1 ? 'reference' : 'references'}
              </span>
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {message.citations.map((citation: Citation) => (
                <button
                  key={citation.id}
                  type="button"
                  onClick={() =>
                    onSelectCitation?.({
                      documentId: citation.documentId,
                      filename: citation.filename,
                      pageNumber: citation.pageNumber,
                      excerpt: citation.excerpt,
                    })
                  }
                  className={cn(
                    'flex flex-col items-start gap-1 rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-left transition-colors cursor-pointer',
                    'hover:border-indigo-300 hover:bg-indigo-50/40 dark:border-slate-800 dark:bg-[#111827] dark:hover:border-indigo-500/40 dark:hover:bg-indigo-950/20',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500'
                  )}
                >
                  <div className="flex w-full items-center justify-between gap-2">
                    <span className="flex min-w-0 items-center gap-1.5 text-xs font-medium text-slate-900 dark:text-slate-100">
                      <FileText className="h-3.5 w-3.5 shrink-0 text-indigo-500 dark:text-indigo-400" />
                      <span className="truncate">{citation.filename}</span>
                    </span>
                    <span className="shrink-0 font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                      Page {citation.pageNumber}
                    </span>
                  </div>
                  <p className="line-clamp-2 text-[12px] leading-4.5 text-slate-500 dark:text-slate-400">
                    {citation.excerpt}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Action toolbar */}
        {!isStreaming && (
          <div className="mt-3 flex items-center gap-1">
            <button
              type="button"
              onClick={handleCopy}
              aria-label="Copy answer"
              className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-xs text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/80 dark:hover:text-slate-100 cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-500" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>

            {isLastAssistant && onRegenerate && (
              <button
                type="button"
                onClick={onRegenerate}
                disabled={disabledActions}
                aria-label="Regenerate answer"
                className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-xs text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 disabled:opacity-40 dark:text-slate-400 dark:hover:bg-slate-800/80 dark:hover:text-slate-100 cursor-pointer"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>Regenerate</span>
              </button>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
