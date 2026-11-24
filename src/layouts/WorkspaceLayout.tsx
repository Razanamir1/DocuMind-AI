import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { PdfPreviewModal } from '../components/PdfPreviewModal';
import { Sidebar } from '../components/Sidebar';
import { api } from '../services/api';
import type { PdfPreviewTarget } from '../types/api';

export function WorkspaceLayout() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [pdfPreviewTarget, setPdfPreviewTarget] = useState<PdfPreviewTarget | null>(null);

  const { data: conversations = [] } = useQuery({
    queryKey: ['conversations'],
    queryFn: () => api.listConversations(),
  });

  const { data: documents = [] } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.listDocuments(),
  });

  const documentsReadyCount = documents.filter((d) => d.status === 'ready').length;

  const handleNewChat = () => {
    navigate('/');
  };

  const handleRenameConversation = async (id: string, title: string) => {
    try {
      await api.updateConversation(id, { title });
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      queryClient.invalidateQueries({ queryKey: ['conversation', id] });
    } catch {
      toast.error('Unable to rename conversation.');
    }
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      await api.deleteConversation(id);
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      if (location.pathname === `/chat/${id}`) {
        navigate('/');
      }
      toast.success('Conversation deleted.');
    } catch {
      toast.error('Unable to delete conversation.');
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-900 dark:bg-[#0B0F19] dark:text-slate-100">
      <Sidebar
        conversations={conversations}
        documentsReadyCount={documentsReadyCount}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        mobileOpen={mobileNavOpen}
        onCloseMobile={() => setMobileNavOpen(false)}
        onNewChat={handleNewChat}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
      />

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Outlet
          context={{
            onOpenMobileNav: () => setMobileNavOpen(true),
            onOpenPdfPreview: (target: PdfPreviewTarget) => setPdfPreviewTarget(target),
          }}
        />
      </div>

      <PdfPreviewModal
        target={pdfPreviewTarget}
        onClose={() => setPdfPreviewTarget(null)}
      />
    </div>
  );
}
