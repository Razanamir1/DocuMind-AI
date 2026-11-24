import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Check, Menu, Moon, RotateCcw, Server, Sliders, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { toast } from 'sonner';
import { Button } from '../components/ui/Button';
import { useTheme } from '../contexts/ThemeContext';
import { cn } from '../lib/utils';
import { api, ApiClientError } from '../services/api';
import type { WorkspaceSettings } from '../types/api';

interface WorkspaceOutletContext {
  onOpenMobileNav: () => void;
}

const DEFAULT_DRAFT: WorkspaceSettings = {
  topK: 5,
  minRelevanceScore: 0.34,
  chunkSize: 850,
  chunkOverlap: 160,
  streamResponses: true,
};

export function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const queryClient = useQueryClient();
  const { onOpenMobileNav } = useOutletContext<WorkspaceOutletContext>();

  const { data, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: () => api.getSettings(),
  });

  const [draft, setDraft] = useState<WorkspaceSettings>(DEFAULT_DRAFT);

  useEffect(() => {
    if (data?.settings) {
      setDraft(data.settings);
    }
  }, [data?.settings]);

  const saveMutation = useMutation({
    mutationFn: (next: Partial<WorkspaceSettings>) => api.updateSettings(next),
    onSuccess: (res) => {
      setDraft(res.settings);
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      toast.success('RAG retrieval settings saved.');
    },
    onError: (err) => {
      const msg = err instanceof ApiClientError ? err.message : 'Unable to save settings.';
      toast.error(msg);
    },
  });

  const handleReset = () => {
    setDraft(DEFAULT_DRAFT);
    saveMutation.mutate(DEFAULT_DRAFT);
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-slate-50 dark:bg-[#0B0F19]">
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
            Settings
          </h1>
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        <div className="mx-auto max-w-3xl space-y-8">
          <div>
            <h2 className="font-display text-xl font-semibold tracking-tight text-slate-950 dark:text-slate-50">
              Workspace & Retrieval Configuration
            </h2>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
              Customize interface appearance and fine-tune how DocuMind AI retrieves and grounds
              PDF passages.
            </p>
          </div>

          {/* Appearance */}
          <section className="rounded-2xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-[#111827]">
            <h3 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
              Appearance
            </h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Choose your preferred workspace theme.
            </p>

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <button
                type="button"
                onClick={() => setTheme('dark')}
                className={cn(
                  'flex items-center justify-between rounded-xl border p-4 text-left transition-colors cursor-pointer',
                  theme === 'dark'
                    ? 'border-indigo-500 bg-indigo-50/40 dark:bg-indigo-500/10'
                    : 'border-slate-200 hover:border-slate-300 dark:border-slate-800 dark:hover:border-slate-700'
                )}
              >
                <div className="flex items-center gap-3">
                  <Moon className="h-4 w-4 text-indigo-500" />
                  <div>
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                      Midnight Dark
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Deep navy canvas optimized for long research sessions
                    </p>
                  </div>
                </div>
                {theme === 'dark' && <Check className="h-4 w-4 shrink-0 text-indigo-500" />}
              </button>

              <button
                type="button"
                onClick={() => setTheme('light')}
                className={cn(
                  'flex items-center justify-between rounded-xl border p-4 text-left transition-colors cursor-pointer',
                  theme === 'light'
                    ? 'border-indigo-500 bg-indigo-50/40 dark:bg-indigo-500/10'
                    : 'border-slate-200 hover:border-slate-300 dark:border-slate-800 dark:hover:border-slate-700'
                )}
              >
                <div className="flex items-center gap-3">
                  <Sun className="h-4 w-4 text-amber-500" />
                  <div>
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                      Daylight Alabaster
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Crisp high-contrast surfaces for bright environments
                    </p>
                  </div>
                </div>
                {theme === 'light' && <Check className="h-4 w-4 shrink-0 text-indigo-500" />}
              </button>
            </div>
          </section>

          {/* RAG Pipeline Parameters */}
          <section className="rounded-2xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-[#111827]">
            <div className="flex items-center gap-2">
              <Sliders className="h-4 w-4 text-indigo-500" />
              <h3 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                RAG Retrieval & Chunking Parameters
              </h3>
            </div>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Controls how many evidence chunks are retrieved and the minimum relevance required
              before generating a document-grounded answer.
            </p>

            {isLoading ? (
              <div className="mt-6 space-y-4 animate-pulse">
                <div className="h-10 rounded bg-slate-200 dark:bg-slate-800" />
                <div className="h-10 rounded bg-slate-200 dark:bg-slate-800" />
              </div>
            ) : (
              <div className="mt-6 space-y-5">
                <div>
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <label
                      htmlFor="setting-topk"
                      className="font-medium text-slate-700 dark:text-slate-300"
                    >
                      Top-K Retrieved Passages
                    </label>
                    <span className="font-mono text-slate-900 dark:text-slate-100 tabular-nums">
                      {draft.topK} chunks
                    </span>
                  </div>
                  <input
                    id="setting-topk"
                    type="range"
                    min={1}
                    max={10}
                    step={1}
                    value={draft.topK}
                    onChange={(e) => setDraft({ ...draft, topK: Number(e.target.value) })}
                    className="w-full accent-indigo-600"
                  />
                </div>

                <div>
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <label
                      htmlFor="setting-threshold"
                      className="font-medium text-slate-700 dark:text-slate-300"
                    >
                      Minimum Evidence Relevance Threshold
                    </label>
                    <span className="font-mono text-slate-900 dark:text-slate-100 tabular-nums">
                      {Math.round(draft.minRelevanceScore * 100)}%
                    </span>
                  </div>
                  <input
                    id="setting-threshold"
                    type="range"
                    min={0.15}
                    max={0.75}
                    step={0.01}
                    value={draft.minRelevanceScore}
                    onChange={(e) =>
                      setDraft({ ...draft, minRelevanceScore: Number(e.target.value) })
                    }
                    className="w-full accent-indigo-600"
                  />
                  <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                    Higher thresholds enforce stricter hallucination prevention and decline
                    weakly supported queries.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div>
                    <label
                      htmlFor="setting-chunk-size"
                      className="mb-1.5 block text-xs font-medium text-slate-700 dark:text-slate-300"
                    >
                      Chunk Size (characters)
                    </label>
                    <input
                      id="setting-chunk-size"
                      type="number"
                      min={300}
                      max={2000}
                      step={50}
                      value={draft.chunkSize}
                      onChange={(e) =>
                        setDraft({ ...draft, chunkSize: Number(e.target.value) })
                      }
                      className="h-9 w-full rounded-lg border border-slate-200 bg-white px-3 font-mono text-sm text-slate-900 tabular-nums focus:border-indigo-400 focus:outline-none dark:border-slate-800 dark:bg-[#0B0F19] dark:text-slate-100"
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="setting-chunk-overlap"
                      className="mb-1.5 block text-xs font-medium text-slate-700 dark:text-slate-300"
                    >
                      Chunk Overlap (characters)
                    </label>
                    <input
                      id="setting-chunk-overlap"
                      type="number"
                      min={0}
                      max={400}
                      step={20}
                      value={draft.chunkOverlap}
                      onChange={(e) =>
                        setDraft({ ...draft, chunkOverlap: Number(e.target.value) })
                      }
                      className="h-9 w-full rounded-lg border border-slate-200 bg-white px-3 font-mono text-sm text-slate-900 tabular-nums focus:border-indigo-400 focus:outline-none dark:border-slate-800 dark:bg-[#0B0F19] dark:text-slate-100"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-end gap-2.5 border-t border-slate-100 pt-4 dark:border-slate-800">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleReset}
                    disabled={saveMutation.isPending}
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    <span>Reset Defaults</span>
                  </Button>
                  <Button
                    variant="primary"
                    size="sm"
                    disabled={saveMutation.isPending}
                    onClick={() => saveMutation.mutate(draft)}
                  >
                    <span>{saveMutation.isPending ? 'Saving…' : 'Save Settings'}</span>
                  </Button>
                </div>
              </div>
            )}
          </section>

          {/* Read-only Server Provider Runtime Summary (No API Key Inputs) */}
          <section className="rounded-2xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-[#111827]">
            <div className="flex items-center gap-2">
              <Server className="h-4 w-4 text-indigo-500" />
              <h3 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                Server AI Runtime
              </h3>
            </div>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              All model credentials remain strictly isolated on the backend server and are never
              transmitted to the browser.
            </p>

            <dl className="mt-4 divide-y divide-slate-100 text-sm dark:divide-slate-800">
              <div className="flex items-center justify-between py-2.5">
                <dt className="text-xs text-slate-500 dark:text-slate-400">Active Provider</dt>
                <dd className="font-medium text-slate-900 dark:text-slate-100">
                  {data?.provider.name ?? 'Google Gemini'}
                </dd>
              </div>
              <div className="flex items-center justify-between py-2.5">
                <dt className="text-xs text-slate-500 dark:text-slate-400">Chat Model</dt>
                <dd className="font-mono text-xs text-slate-700 dark:text-slate-300">
                  {data?.provider.chatModel ?? 'gemini-4-hydrogen'}
                </dd>
              </div>
              <div className="flex items-center justify-between py-2.5">
                <dt className="text-xs text-slate-500 dark:text-slate-400">Embedding Model</dt>
                <dd className="font-mono text-xs text-slate-700 dark:text-slate-300">
                  {data?.provider.embeddingModel ?? 'gemini-embedding-2'}
                </dd>
              </div>
              <div className="flex items-center justify-between py-2.5">
                <dt className="text-xs text-slate-500 dark:text-slate-400">Backend & Local Storage</dt>
                <dd className="text-xs text-slate-700 dark:text-slate-300">
                  {data?.provider.backendEngine ?? 'Python FastAPI + ChromaDB'} · SQLite (SQLAlchemy)
                </dd>
              </div>
              <div className="flex items-center justify-between gap-4 py-2.5">
                <dt className="shrink-0 text-xs text-slate-500 dark:text-slate-400">
                  Storage Durability
                </dt>
                <dd className="text-right text-xs text-slate-600 dark:text-slate-400">
                  Instance-local disk (ephemeral across Cloud Run restarts or scaling)
                </dd>
              </div>
            </dl>
          </section>
        </div>
      </main>
    </div>
  );
}
