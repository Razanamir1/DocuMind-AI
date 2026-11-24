import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertCircle,
  CheckCircle2,
  Database,
  Eye,
  FileText,
  Loader2,
  Menu,
  MessageSquare,
  Search,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { toast } from 'sonner';
import { Button } from '../components/ui/Button';
import { cn, formatBytes, formatRelativeDate } from '../lib/utils';
import { api, ApiClientError } from '../services/api';
import type { DocumentItem, PdfPreviewTarget } from '../types/api';

interface WorkspaceOutletContext {
  onOpenMobileNav: () => void;
  onOpenPdfPreview: (target: PdfPreviewTarget) => void;
}

type StatusFilter = 'all' | 'ready' | 'processing' | 'failed';

interface UploadProgressState {
  filename: string;
  percent: number;
  stage: string;
}

export function KnowledgeBasePage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { onOpenMobileNav, onOpenPdfPreview } = useOutletContext<WorkspaceOutletContext>();

  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadState, setUploadState] = useState<UploadProgressState | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const { data: allDocuments = [], isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.listDocuments(),
  });

  const filteredDocuments = allDocuments.filter((doc) => {
    const matchesSearch =
      !searchQuery.trim() ||
      doc.filename.toLowerCase().includes(searchQuery.trim().toLowerCase());
    const matchesStatus = statusFilter === 'all' || doc.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const totalPages = allDocuments.reduce((sum, d) => sum + d.pageCount, 0);
  const totalChunks = allDocuments.reduce((sum, d) => sum + d.chunkCount, 0);

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteDocument(id),
    onMutate: (id) => setDeletingId(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['health'] });
      toast.success('Document removed from Knowledge Base and active retrieval.');
    },
    onError: (err) => {
      const msg = err instanceof ApiClientError ? err.message : 'Failed to delete document.';
      toast.error(msg);
    },
    onSettled: () => setDeletingId(null),
  });

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0 || uploadState) return;
    const file = files[0];

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      toast.error('Unsupported file type. Please upload a valid PDF (.pdf) document.');
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      toast.error('File exceeds the 25 MB maximum upload limit.');
      return;
    }

    setUploadState({
      filename: file.name,
      percent: 12,
      stage: 'Uploading PDF binary…',
    });

    try {
      const doc = await api.uploadDocument(file, (pct) => {
        const stage =
          pct < 45
            ? 'Uploading PDF binary…'
            : pct < 85
              ? 'Extracting pages & splitting overlapping chunks…'
              : 'Generating embeddings & indexing vector store…';
        setUploadState({ filename: file.name, percent: pct, stage });
      });

      queryClient.setQueryData<DocumentItem[]>(['documents'], (prev) =>
        prev ? [doc, ...prev.filter((item) => item.id !== doc.id)] : [doc]
      );
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['health'] });
      toast.success(
        `Indexed "${doc.filename}" (${doc.pageCount} ${doc.pageCount === 1 ? 'page' : 'pages'}, ${doc.chunkCount} chunks).`
      );
    } catch (err) {
      const message =
        err instanceof ApiClientError
          ? err.message
          : 'Failed to process this PDF document.';
      toast.error(message);
    } finally {
      setUploadState(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
  };

  const renderStatus = (doc: DocumentItem) => {
    if (doc.status === 'ready') {
      if (doc.embeddingStatus === 'pending') {
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-600 dark:text-amber-400">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            <span>Pending index</span>
          </span>
        );
      }
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700 dark:text-emerald-400">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
          <span>Ready</span>
        </span>
      );
    }
    if (doc.status === 'processing' || doc.status === 'uploading') {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-indigo-600 dark:text-indigo-400">
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
          <span>{doc.status === 'uploading' ? 'Uploading' : 'Processing'}</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-red-600 dark:text-red-400">
        <AlertCircle className="h-3.5 w-3.5 shrink-0" />
        <span>Failed</span>
      </span>
    );
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-slate-50 dark:bg-[#0B0F19]">
      {/* Header */}
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-slate-200/80 bg-white/80 px-4 backdrop-blur-xs dark:border-slate-800/80 dark:bg-[#0B0F19]/90 sm:px-6">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={onOpenMobileNav}
            aria-label="Open navigation menu"
            className="h-8 w-8 md:hidden"
          >
            <Menu className="h-4 w-4" />
          </Button>
          <h1 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
            Knowledge Base
          </h1>
        </div>

        <Button
          variant="primary"
          size="sm"
          disabled={Boolean(uploadState)}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadCloud className="h-3.5 w-3.5" />
          <span>Upload PDF</span>
        </Button>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        <div className="mx-auto max-w-5xl space-y-6">
          {/* Page Intro & Unboxed Inline Stats */}
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
            <div>
              <h2 className="font-display text-xl font-semibold tracking-tight text-slate-950 dark:text-slate-50">
                Indexed PDF Documents
              </h2>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                Upload PDFs to parse pages, create overlapping chunks, and index them for
                citation-grounded RAG retrieval.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 tabular-nums">
              <span>
                {allDocuments.length} {allDocuments.length === 1 ? 'document' : 'documents'}
              </span>
              <span aria-hidden="true">·</span>
              <span>{totalPages} pages</span>
              <span aria-hidden="true">·</span>
              <span>{totalChunks} indexed chunks</span>
            </div>
          </div>

          {/* Drag-and-Drop PDF Upload Area */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            className={cn(
              'rounded-2xl border border-dashed p-6 transition-all duration-150',
              isDragOver
                ? 'border-indigo-500 bg-indigo-50/70 dark:border-indigo-400 dark:bg-indigo-950/30'
                : 'border-slate-300 bg-white hover:border-slate-400 dark:border-slate-700/90 dark:bg-[#111827] dark:hover:border-slate-600'
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleInputChange}
              aria-label="Select PDF document to upload"
              className="hidden"
            />

            {uploadState ? (
              <div className="mx-auto max-w-lg py-2">
                <div className="mb-2 flex items-center justify-between gap-3 text-sm">
                  <span className="flex min-w-0 items-center gap-2 font-medium text-slate-900 dark:text-slate-100">
                    <Loader2 className="h-4 w-4 shrink-0 animate-spin text-indigo-500" />
                    <span className="truncate">{uploadState.filename}</span>
                  </span>
                  <span className="font-mono text-xs text-indigo-600 dark:text-indigo-400 tabular-nums">
                    {uploadState.percent}%
                  </span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                  <div
                    className="h-full rounded-full bg-indigo-600 transition-all duration-300 dark:bg-indigo-500"
                    style={{ width: `${uploadState.percent}%` }}
                  />
                </div>
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  {uploadState.stage}
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
                <div className="flex items-center gap-4">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 dark:bg-indigo-500/15 dark:text-indigo-400">
                    <UploadCloud className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                      Drag and drop a PDF file here, or browse your computer
                    </p>
                    <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                      PDF files with extractable text up to 25 MB · Scanned image-only PDFs are
                      detected automatically (OCR not enabled)
                    </p>
                  </div>
                </div>

                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="shrink-0"
                >
                  Browse Files
                </Button>
              </div>
            )}
          </div>

          {/* Search & Segmented Status Filter */}
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
            <div className="relative flex-1 sm:max-w-sm">
              <Search className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search documents by filename…"
                aria-label="Search documents"
                className="h-9 w-full rounded-lg border border-slate-200 bg-white pr-3 pl-9 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none dark:border-slate-800 dark:bg-[#111827] dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-indigo-500"
              />
            </div>

            <div
              role="tablist"
              aria-label="Filter documents by status"
              className="inline-flex items-center rounded-lg border border-slate-200/70 bg-slate-100 p-0.5 dark:border-slate-800 dark:bg-slate-900"
            >
              {(['all', 'ready', 'processing', 'failed'] as StatusFilter[]).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  role="tab"
                  aria-selected={statusFilter === tab}
                  onClick={() => setStatusFilter(tab)}
                  className={cn(
                    'rounded-md px-3 py-1 text-xs font-medium capitalize whitespace-nowrap transition-colors cursor-pointer',
                    statusFilter === tab
                      ? 'bg-white text-slate-950 shadow-2xs dark:bg-slate-800 dark:text-white'
                      : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                  )}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          {/* Document Table */}
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#111827]">
            {isLoading ? (
              <div className="divide-y divide-slate-100 p-4 dark:divide-slate-800">
                {[1, 2, 3].map((row) => (
                  <div key={row} className="flex items-center justify-between py-3.5 animate-pulse">
                    <div className="h-4 w-56 rounded bg-slate-200 dark:bg-slate-800" />
                    <div className="h-4 w-32 rounded bg-slate-200 dark:bg-slate-800" />
                  </div>
                ))}
              </div>
            ) : filteredDocuments.length === 0 ? (
              <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-slate-100 text-slate-400 dark:bg-slate-800 dark:text-slate-500">
                  <Database className="h-5 w-5" />
                </div>
                <h3 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  {allDocuments.length === 0
                    ? 'No documents indexed yet'
                    : 'No documents match your filter'}
                </h3>
                <p className="mt-1 max-w-sm text-xs leading-5 text-slate-500 dark:text-slate-400">
                  {allDocuments.length === 0
                    ? 'Upload your first PDF document above to enable evidence-grounded answers with page citations.'
                    : 'Try clearing your search term or selecting a different status filter.'}
                </p>
                {allDocuments.length === 0 ? (
                  <Button
                    size="sm"
                    className="mt-4"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Upload First PDF
                  </Button>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-4"
                    onClick={() => {
                      setSearchQuery('');
                      setStatusFilter('all');
                    }}
                  >
                    Reset Filters
                  </Button>
                )}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/70 text-xs font-medium text-slate-500 dark:border-slate-800 dark:bg-slate-900/50 dark:text-slate-400">
                      <th className="px-5 py-3">Document</th>
                      <th className="px-4 py-3 text-right">Size</th>
                      <th className="px-4 py-3 text-right">Pages</th>
                      <th className="px-4 py-3 text-right">Chunks</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3">Uploaded</th>
                      <th className="px-5 py-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                    {filteredDocuments.map((doc) => (
                      <tr
                        key={doc.id}
                        className="transition-colors hover:bg-slate-50/80 dark:hover:bg-slate-800/40"
                      >
                        <td className="max-w-[400px] px-5 py-3.5">
                          <div className="flex items-center gap-3">
                            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-indigo-600 dark:bg-slate-800 dark:text-indigo-400">
                              <FileText className="h-4 w-4" />
                            </div>
                            <div className="min-w-0">
                              <button
                                type="button"
                                onClick={() =>
                                  onOpenPdfPreview({
                                    documentId: doc.id,
                                    filename: doc.filename,
                                    pageNumber: 1,
                                  })
                                }
                                className="block max-w-full truncate font-medium text-slate-900 hover:text-indigo-600 dark:text-slate-100 dark:hover:text-indigo-400 cursor-pointer"
                              >
                                {doc.filename}
                              </button>
                              <p className="text-[11px] text-slate-400 dark:text-slate-500">
                                {doc.embeddingStatus === 'vector'
                                  ? 'ChromaDB vector index'
                                  : 'Pending vector index'}
                              </p>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3.5 text-right font-mono text-xs text-slate-600 dark:text-slate-400 tabular-nums whitespace-nowrap">
                          {formatBytes(doc.fileSizeBytes)}
                        </td>
                        <td className="px-4 py-3.5 text-right font-mono text-xs text-slate-600 dark:text-slate-400 tabular-nums">
                          {doc.pageCount}
                        </td>
                        <td className="px-4 py-3.5 text-right font-mono text-xs text-slate-600 dark:text-slate-400 tabular-nums">
                          {doc.chunkCount}
                        </td>
                        <td className="px-4 py-3.5 whitespace-nowrap">{renderStatus(doc)}</td>
                        <td className="px-4 py-3.5 text-xs text-slate-500 dark:text-slate-400 tabular-nums whitespace-nowrap">
                          {formatRelativeDate(doc.createdAt)}
                        </td>
                        <td className="px-5 py-3.5 text-right whitespace-nowrap">
                          <div className="inline-flex items-center gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              aria-label={`Preview ${doc.filename}`}
                              title="Preview PDF"
                              onClick={() =>
                                onOpenPdfPreview({
                                  documentId: doc.id,
                                  filename: doc.filename,
                                  pageNumber: 1,
                                })
                              }
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              aria-label={`Chat with ${doc.filename}`}
                              title="Ask questions about this document"
                              onClick={() => navigate(`/?mode=documents&doc=${encodeURIComponent(doc.id)}`)}
                            >
                              <MessageSquare className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-slate-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/50 dark:hover:text-red-400"
                              disabled={deletingId === doc.id}
                              aria-label={`Delete ${doc.filename}`}
                              title="Remove from Knowledge Base"
                              onClick={() => deleteMutation.mutate(doc.id)}
                            >
                              {deletingId === doc.id ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Trash2 className="h-4 w-4" />
                              )}
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
