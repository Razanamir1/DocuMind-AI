import {
  ChevronLeft,
  ChevronRight,
  Download,
  FileText,
  Highlighter,
  Loader2,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import * as pdfjsLib from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { useEffect, useRef, useState } from 'react';
import { api } from '../services/api';
import type { PdfPreviewTarget } from '../types/api';
import { Button } from './ui/Button';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

interface PdfPreviewModalProps {
  target: PdfPreviewTarget | null;
  onClose: () => void;
}

function highlightExcerptInText(pageText: string, excerpt?: string) {
  if (!excerpt || !pageText) return pageText;

  // Clean ellipsis and search for longest meaningful phrase from excerpt
  const cleanExcerpt = excerpt.replace(/^[….\s]+|[….\s]+$/g, '').trim();
  const candidates = [
    cleanExcerpt.slice(0, 90),
    cleanExcerpt.slice(0, 50),
    ...cleanExcerpt
      .split(/[.!?]/)
      .map((s) => s.trim())
      .filter((s) => s.length >= 24),
  ].filter(Boolean);

  const lowerPage = pageText.toLowerCase();
  for (const snippet of candidates) {
    const idx = lowerPage.indexOf(snippet.toLowerCase());
    if (idx !== -1) {
      const before = pageText.slice(0, idx);
      const matched = pageText.slice(idx, idx + snippet.length);
      const after = pageText.slice(idx + snippet.length);
      return (
        <>
          {before}
          <mark className="rounded bg-indigo-500/25 px-1 py-0.5 text-slate-950 dark:bg-indigo-500/35 dark:text-white">
            {matched}
          </mark>
          {after}
        </>
      );
    }
  }

  return pageText;
}

export function PdfPreviewModal({ target, onClose }: PdfPreviewModalProps) {
  const [pageNumber, setPageNumber] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [viewMode, setViewMode] = useState<'rendered' | 'text'>('rendered');
  const [loadingPdf, setLoadingPdf] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [pageText, setPageText] = useState<string>('');
  const [loadingText, setLoadingText] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pdfDocRef = useRef<pdfjsLib.PDFDocumentProxy | null>(null);
  const loadingTaskRef = useRef<pdfjsLib.PDFDocumentLoadingTask | null>(null);

  useEffect(() => {
    if (!target) return;
    setPageNumber(Math.max(1, target.pageNumber));
    setScale(1.0);
    setViewMode('rendered');
    setPdfError(null);
  }, [target]);

  // Close on Escape
  useEffect(() => {
    if (!target) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [target, onClose]);

  // Load PDF binary into pdf.js
  useEffect(() => {
    if (!target) return;
    let cancelled = false;

    async function loadPdf() {
      setLoadingPdf(true);
      setPdfError(null);
      try {
        if (loadingTaskRef.current) {
          await loadingTaskRef.current.destroy();
          loadingTaskRef.current = null;
          pdfDocRef.current = null;
        }
        const url = api.getDocumentFileUrl(target!.documentId);
        const task = pdfjsLib.getDocument({ url });
        loadingTaskRef.current = task;
        const doc = await task.promise;
        if (cancelled) {
          await task.destroy();
          return;
        }
        pdfDocRef.current = doc;
        setTotalPages(doc.numPages);
        setPageNumber((prev) => Math.min(Math.max(1, prev), doc.numPages));
      } catch {
        if (!cancelled) {
          setPdfError(
            'Unable to render the PDF binary for this document. It may have been removed from the Knowledge Base.'
          );
        }
      } finally {
        if (!cancelled) setLoadingPdf(false);
      }
    }

    loadPdf();

    return () => {
      cancelled = true;
    };
  }, [target]);

  // Render active page onto canvas
  useEffect(() => {
    if (!target || viewMode !== 'rendered' || loadingPdf || pdfError) return;
    const doc = pdfDocRef.current;
    const canvas = canvasRef.current;
    if (!doc || !canvas) return;

    let renderTask: pdfjsLib.RenderTask | null = null;
    let cancelled = false;

    async function renderPage() {
      try {
        const safePage = Math.min(Math.max(1, pageNumber), doc!.numPages);
        const page = await doc!.getPage(safePage);
        if (cancelled) return;

        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const viewport = page.getViewport({ scale });
        const context = canvas!.getContext('2d');
        if (!context) return;

        canvas!.width = Math.floor(viewport.width * dpr);
        canvas!.height = Math.floor(viewport.height * dpr);
        canvas!.style.width = `${Math.floor(viewport.width)}px`;
        canvas!.style.height = `${Math.floor(viewport.height)}px`;

        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        renderTask = page.render({ canvas: canvas!, canvasContext: context, viewport });
        await renderTask.promise;
      } catch {
        // Ignore cancelled renders on rapid page switches
      }
    }

    renderPage();

    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
  }, [target, pageNumber, scale, viewMode, loadingPdf, pdfError]);

  // Fetch extracted text for the current page (for highlighting & text view)
  useEffect(() => {
    if (!target) return;
    let cancelled = false;

    async function loadPageText() {
      setLoadingText(true);
      try {
        const res = await api.getDocumentPage(target!.documentId, pageNumber);
        if (!cancelled) {
          setPageText(res.page.content);
          setTotalPages(res.document.pageCount);
        }
      } catch {
        if (!cancelled) setPageText('');
      } finally {
        if (!cancelled) setLoadingText(false);
      }
    }

    loadPageText();
    return () => {
      cancelled = true;
    };
  }, [target, pageNumber]);

  return (
    <AnimatePresence>
      {target && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`PDF preview: ${target.filename}`}
          className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6"
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            onClick={onClose}
            className="absolute inset-0 bg-slate-950/70 backdrop-blur-[2px]"
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.98, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: 10 }}
            transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            className="relative z-10 flex h-[88vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-[#0F1524]"
          >
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-3.5 dark:border-slate-800">
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600 dark:bg-indigo-500/15 dark:text-indigo-400">
                  <FileText className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <h2 className="truncate font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                    {target.filename}
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400 tabular-nums">
                    Page {pageNumber} of {totalPages}
                    {target.excerpt ? ' · Cited source passage' : ''}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* View mode toggle */}
                <div
                  role="tablist"
                  aria-label="Preview display mode"
                  className="flex items-center rounded-lg bg-slate-100 p-0.5 dark:bg-slate-800/90"
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={viewMode === 'rendered'}
                    onClick={() => setViewMode('rendered')}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                      viewMode === 'rendered'
                        ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-950 dark:text-white'
                        : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                    }`}
                  >
                    PDF Page
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={viewMode === 'text'}
                    onClick={() => setViewMode('text')}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                      viewMode === 'text'
                        ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-950 dark:text-white'
                        : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                    }`}
                  >
                    Highlighted Text
                  </button>
                </div>

                <a
                  href={api.getDocumentFileUrl(target.documentId)}
                  download={target.filename}
                  aria-label={`Download ${target.filename}`}
                  className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50 dark:border-slate-800 dark:text-slate-300 dark:hover:bg-slate-800 whitespace-nowrap"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Download</span>
                </a>

                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClose}
                  aria-label="Close PDF preview"
                  className="h-8 w-8"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Body: Main viewer + Evidence sidebar */}
            <div className="flex min-h-0 min-w-0 flex-1 flex-col lg:flex-row">
              {/* Main PDF / Highlighted Text pane */}
              <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-slate-100/80 dark:bg-[#090D16]">
                {/* Page & Zoom toolbar */}
                <div className="flex items-center justify-between border-b border-slate-200/80 bg-white/80 px-4 py-2 dark:border-slate-800/80 dark:bg-[#0F1524]/90">
                  <div className="flex items-center gap-1.5">
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-7 w-7"
                      disabled={pageNumber <= 1}
                      onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
                      aria-label="Previous page"
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                    </Button>
                    <span className="px-2 font-mono text-xs text-slate-700 dark:text-slate-300 tabular-nums">
                      {pageNumber} / {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-7 w-7"
                      disabled={pageNumber >= totalPages}
                      onClick={() => setPageNumber((p) => Math.min(totalPages, p + 1))}
                      aria-label="Next page"
                    >
                      <ChevronRight className="h-3.5 w-3.5" />
                    </Button>
                  </div>

                  {viewMode === 'rendered' && (
                    <div className="flex items-center gap-1.5">
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-7 w-7"
                        disabled={scale <= 0.7}
                        onClick={() => setScale((s) => Math.max(0.7, Number((s - 0.15).toFixed(2))))}
                        aria-label="Zoom out"
                      >
                        <ZoomOut className="h-3.5 w-3.5" />
                      </Button>
                      <span className="w-12 text-center font-mono text-xs text-slate-600 dark:text-slate-400 tabular-nums">
                        {Math.round(scale * 100)}%
                      </span>
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-7 w-7"
                        disabled={scale >= 2}
                        onClick={() => setScale((s) => Math.min(2, Number((s + 0.15).toFixed(2))))}
                        aria-label="Zoom in"
                      >
                        <ZoomIn className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  )}
                </div>

                {/* Viewport */}
                <div className="flex-1 overflow-auto p-5">
                  {viewMode === 'rendered' ? (
                    <div className="flex min-h-full items-start justify-center">
                      {loadingPdf ? (
                        <div className="flex flex-col items-center gap-2.5 py-20 text-slate-500 dark:text-slate-400">
                          <Loader2 className="h-5 w-5 animate-spin text-indigo-500" />
                          <span className="text-xs">Rendering PDF page {pageNumber}…</span>
                        </div>
                      ) : pdfError ? (
                        <div className="max-w-md rounded-xl border border-red-200 bg-red-50/70 p-4 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300">
                          {pdfError}
                        </div>
                      ) : (
                        <canvas
                          ref={canvasRef}
                          className="rounded-lg border border-slate-200 bg-white shadow-md dark:border-slate-700"
                        />
                      )}
                    </div>
                  ) : (
                    <div className="mx-auto max-w-2xl rounded-xl border border-slate-200 bg-white p-6 shadow-xs dark:border-slate-800 dark:bg-[#111827]">
                      {loadingText ? (
                        <div className="space-y-3 animate-pulse">
                          <div className="h-4 w-2/3 rounded bg-slate-200 dark:bg-slate-800" />
                          <div className="h-3.5 w-full rounded bg-slate-200 dark:bg-slate-800" />
                          <div className="h-3.5 w-5/6 rounded bg-slate-200 dark:bg-slate-800" />
                        </div>
                      ) : pageText ? (
                        <div className="whitespace-pre-wrap text-sm leading-7 text-slate-800 dark:text-slate-200">
                          {highlightExcerptInText(pageText, target.excerpt)}
                        </div>
                      ) : (
                        <p className="text-sm text-slate-500 dark:text-slate-400">
                          No extracted text available for page {pageNumber}.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Supporting Excerpt Sidebar */}
              {target.excerpt && (
                <aside className="flex w-full shrink-0 flex-col overflow-y-auto border-t border-slate-200 bg-white p-5 break-words dark:border-slate-800 dark:bg-[#111827] lg:w-80 lg:border-t-0 lg:border-l">
                  <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-indigo-600 dark:text-indigo-400">
                    <Highlighter className="h-3.5 w-3.5 shrink-0" />
                    <span>Supporting Excerpt</span>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3.5 text-[13.5px] leading-6 text-slate-700 break-words dark:border-slate-800 dark:bg-[#0B0F19] dark:text-slate-300">
                    {target.excerpt}
                  </div>
                  <p className="mt-3 text-xs leading-5 text-slate-500 break-words dark:text-slate-400">
                    Extracted directly from{' '}
                    <span className="font-medium text-slate-700 break-all dark:text-slate-300">
                      {target.filename}
                    </span>{' '}
                    on page{' '}
                    <span className="font-mono tabular-nums">{target.pageNumber}</span>. Switch to{' '}
                    <button
                      type="button"
                      onClick={() => setViewMode('text')}
                      className="font-medium text-indigo-600 underline underline-offset-2 hover:text-indigo-500 dark:text-indigo-400 cursor-pointer"
                    >
                      Highlighted Text
                    </button>{' '}
                    to inspect the passage in full page context.
                  </p>
                </aside>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
