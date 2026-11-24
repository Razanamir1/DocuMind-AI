import {
  Check,
  Database,
  FileText,
  MessageSquare,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Plus,
  Settings,
  Sun,
  Trash2,
  X,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import { cn } from '../lib/utils';
import type { ConversationItem } from '../types/api';
import { Button } from './ui/Button';
import { Logo } from './ui/Logo';

const USER_AVATAR_URL = '/src/assets/images/avatar_workspace_user_1795361326102.jpg';

interface SidebarProps {
  conversations: ConversationItem[];
  documentsReadyCount: number;
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  onNewChat: () => void;
  onRenameConversation: (id: string, title: string) => Promise<void>;
  onDeleteConversation: (id: string) => Promise<void>;
}

export function Sidebar({
  conversations,
  documentsReadyCount,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
  onNewChat,
  onRenameConversation,
  onDeleteConversation,
}: SidebarProps) {
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [avatarFailed, setAvatarFailed] = useState(false);
  const editInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);

  const startEditing = (convo: ConversationItem) => {
    setEditingId(convo.id);
    setEditTitle(convo.title);
  };

  const commitEdit = async (id: string) => {
    const trimmed = editTitle.trim();
    setEditingId(null);
    if (!trimmed) return;
    await onRenameConversation(id, trimmed);
  };

  const isChatRoute =
    location.pathname === '/' || location.pathname.startsWith('/chat/');

  const sidebarInner = (isMobileDrawer: boolean) => {
    const isCollapsed = !isMobileDrawer && collapsed;

    return (
      <div className="flex h-full flex-col bg-white text-slate-800 dark:bg-[#0E1422] dark:text-slate-200">
        {/* Top brand row */}
        <div
          className={cn(
            'flex h-14 shrink-0 items-center border-b border-slate-200/80 px-3.5 dark:border-slate-800/80',
            isCollapsed ? 'justify-center' : 'justify-between'
          )}
        >
          {!isCollapsed && (
            <button
              type="button"
              onClick={() => {
                onNewChat();
                onCloseMobile();
              }}
              aria-label="DocuMind AI home"
              className="flex min-w-0 items-center rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 cursor-pointer"
            >
              <Logo />
            </button>
          )}

          {isMobileDrawer ? (
            <Button
              variant="ghost"
              size="icon"
              onClick={onCloseMobile}
              aria-label="Close navigation drawer"
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleCollapse}
              aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              className="h-8 w-8"
            >
              {isCollapsed ? (
                <PanelLeftOpen className="h-4 w-4" />
              ) : (
                <PanelLeftClose className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>

        {/* New Chat primary CTA */}
        <div className="p-3">
          <Button
            variant="primary"
            onClick={() => {
              onNewChat();
              onCloseMobile();
            }}
            aria-label="Start new chat"
            className={cn('w-full', isCollapsed ? 'px-0' : 'justify-start px-3.5')}
          >
            <Plus className="h-4 w-4 shrink-0" />
            {!isCollapsed && <span>New Chat</span>}
          </Button>
        </div>

        {/* Primary workspace navigation */}
        <nav aria-label="Primary workspace navigation" className="space-y-0.5 px-2.5">
          <button
            type="button"
            onClick={() => {
              navigate('/');
              onCloseMobile();
            }}
            className={cn(
              'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors cursor-pointer whitespace-nowrap',
              isCollapsed && 'justify-center px-0',
              isChatRoute
                ? 'bg-slate-100 text-slate-950 dark:bg-slate-800/80 dark:text-white'
                : 'text-slate-600 hover:bg-slate-100/70 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/50 dark:hover:text-slate-100'
            )}
          >
            <MessageSquare className="h-4 w-4 shrink-0" />
            {!isCollapsed && <span>Workspace Chat</span>}
          </button>

          <NavLink
            to="/knowledge-base"
            onClick={onCloseMobile}
            className={({ isActive }) =>
              cn(
                'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors whitespace-nowrap',
                isCollapsed && 'justify-center px-0',
                isActive
                  ? 'bg-slate-100 text-slate-950 dark:bg-slate-800/80 dark:text-white'
                  : 'text-slate-600 hover:bg-slate-100/70 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/50 dark:hover:text-slate-100'
              )
            }
          >
            <Database className="h-4 w-4 shrink-0" />
            {!isCollapsed && (
              <span className="flex flex-1 items-center justify-between gap-2">
                <span>Knowledge Base</span>
                <span className="font-mono text-xs text-slate-400 dark:text-slate-500 tabular-nums">
                  {documentsReadyCount}
                </span>
              </span>
            )}
          </NavLink>

          <NavLink
            to="/settings"
            onClick={onCloseMobile}
            className={({ isActive }) =>
              cn(
                'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors whitespace-nowrap',
                isCollapsed && 'justify-center px-0',
                isActive
                  ? 'bg-slate-100 text-slate-950 dark:bg-slate-800/80 dark:text-white'
                  : 'text-slate-600 hover:bg-slate-100/70 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/50 dark:hover:text-slate-100'
              )
            }
          >
            <Settings className="h-4 w-4 shrink-0" />
            {!isCollapsed && <span>Settings</span>}
          </NavLink>
        </nav>

        {/* Conversations list */}
        <div className="mt-4 flex min-h-0 flex-1 flex-col border-t border-slate-200/70 pt-3 dark:border-slate-800/70">
          {!isCollapsed && (
            <div className="mb-1.5 flex items-center justify-between px-4">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
                Conversations
              </span>
              <span className="font-mono text-[11px] text-slate-400 dark:text-slate-500 tabular-nums">
                {conversations.length}
              </span>
            </div>
          )}

          <div className="flex-1 space-y-0.5 overflow-y-auto px-2.5 pb-3">
            {conversations.length === 0 ? (
              !isCollapsed && (
                <p className="px-2 py-4 text-xs text-slate-400 dark:text-slate-500">
                  Your saved conversations will appear here.
                </p>
              )
            ) : (
              conversations.map((convo) => {
                const active = location.pathname === `/chat/${convo.id}`;
                const isEditing = editingId === convo.id;

                if (isCollapsed) {
                  return (
                    <button
                      key={convo.id}
                      type="button"
                      onClick={() => navigate(`/chat/${convo.id}`)}
                      title={convo.title}
                      aria-label={`Open conversation ${convo.title}`}
                      className={cn(
                        'flex h-9 w-full items-center justify-center rounded-lg transition-colors cursor-pointer',
                        active
                          ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-500/15 dark:text-indigo-300'
                          : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/60 dark:hover:text-slate-100'
                      )}
                    >
                      {convo.mode === 'documents' ? (
                        <FileText className="h-4 w-4" />
                      ) : (
                        <MessageSquare className="h-4 w-4" />
                      )}
                    </button>
                  );
                }

                return (
                  <div
                    key={convo.id}
                    className={cn(
                      'group relative flex items-center gap-1 rounded-lg transition-colors',
                      active
                        ? 'bg-slate-100 text-slate-950 dark:bg-slate-800/85 dark:text-white'
                        : 'text-slate-600 hover:bg-slate-100/70 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/50 dark:hover:text-slate-100'
                    )}
                  >
                    {isEditing ? (
                      <form
                        onSubmit={(e) => {
                          e.preventDefault();
                          commitEdit(convo.id);
                        }}
                        className="flex flex-1 items-center gap-1 px-2 py-1"
                      >
                        <input
                          ref={editInputRef}
                          type="text"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Escape') setEditingId(null);
                          }}
                          onBlur={() => commitEdit(convo.id)}
                          aria-label="Rename conversation"
                          className="h-7 min-w-0 flex-1 rounded-md border border-indigo-400 bg-white px-2 text-xs text-slate-900 focus:outline-none dark:border-indigo-500 dark:bg-slate-900 dark:text-white"
                        />
                        <button
                          type="submit"
                          aria-label="Save conversation title"
                          className="flex h-6 w-6 shrink-0 items-center justify-center rounded text-indigo-600 hover:bg-indigo-50 dark:text-indigo-400 dark:hover:bg-indigo-950/60 cursor-pointer"
                        >
                          <Check className="h-3.5 w-3.5" />
                        </button>
                      </form>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            navigate(`/chat/${convo.id}`);
                            onCloseMobile();
                          }}
                          className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-2 text-left text-[13px] font-medium cursor-pointer"
                        >
                          {convo.mode === 'documents' ? (
                            <FileText className="h-3.5 w-3.5 shrink-0 text-indigo-500 dark:text-indigo-400" />
                          ) : (
                            <MessageSquare className="h-3.5 w-3.5 shrink-0 text-slate-400 dark:text-slate-500" />
                          )}
                          <span className="truncate">{convo.title}</span>
                        </button>

                        <div className="flex shrink-0 items-center gap-0.5 pr-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              startEditing(convo);
                            }}
                            aria-label={`Rename "${convo.title}"`}
                            className="flex h-6 w-6 items-center justify-center rounded-md text-slate-400 hover:bg-slate-200/70 hover:text-slate-700 dark:hover:bg-slate-700 dark:hover:text-slate-100 cursor-pointer"
                          >
                            <Pencil className="h-3 w-3" />
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteConversation(convo.id);
                            }}
                            aria-label={`Delete "${convo.title}"`}
                            className="flex h-6 w-6 items-center justify-center rounded-md text-slate-400 hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-950/60 dark:hover:text-red-400 cursor-pointer"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer: Theme toggle + User profile */}
        <div className="shrink-0 border-t border-slate-200/80 p-3 dark:border-slate-800/80">
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            className={cn(
              'mb-2.5 flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/60 dark:hover:text-slate-100 cursor-pointer whitespace-nowrap',
              isCollapsed && 'justify-center px-0'
            )}
          >
            {theme === 'dark' ? (
              <Sun className="h-4 w-4 shrink-0 text-amber-400" />
            ) : (
              <Moon className="h-4 w-4 shrink-0 text-indigo-600" />
            )}
            {!isCollapsed && (
              <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
            )}
          </button>

          <div
            className={cn(
              'flex items-center gap-2.5 rounded-xl px-2 py-1.5',
              isCollapsed && 'justify-center px-0'
            )}
          >
            {!avatarFailed ? (
              <img
                src={USER_AVATAR_URL}
                alt="Workspace architect profile"
                referrerPolicy="no-referrer"
                onError={() => setAvatarFailed(true)}
                className="h-8 w-8 shrink-0 rounded-full object-cover ring-1 ring-slate-200 dark:ring-slate-700"
              />
            ) : (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
                DM
              </div>
            )}

            {!isCollapsed && (
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
                  Workspace Architect
                </p>
                <p className="truncate text-[11px] text-slate-500 dark:text-slate-400">
                  Local Enterprise Edition
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      {/* Desktop collapsible sidebar */}
      <aside
        className={cn(
          'hidden h-full shrink-0 border-r border-slate-200/80 transition-[width] duration-200 ease-out dark:border-slate-800/80 md:block',
          collapsed ? 'w-[68px]' : 'w-[264px]'
        )}
      >
        {sidebarInner(false)}
      </aside>

      {/* Mobile slide-over drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 flex md:hidden">
          <div
            onClick={onCloseMobile}
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-[1px]"
            aria-hidden="true"
          />
          <aside className="relative z-10 h-full w-[280px] max-w-[82vw] border-r border-slate-200 shadow-xl dark:border-slate-800">
            {sidebarInner(true)}
          </aside>
        </div>
      )}
    </>
  );
}
