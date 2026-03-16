import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGoogleDrivePicker, type DriveFile } from "./hooks/useGoogleDrivePicker";
import FolderTree from "./docs/FolderTree";
import DocPreview from "./docs/DocPreview";
import { buildTree, filterTree, buildBreadcrumb } from "./docs/buildTree";
import type { KBProject, KBFolder, KBItem, TreeNode } from "./docs/types";

const MAX_FILE_SIZE = 25 * 1024 * 1024;
const ACCEPTED_FILE_TYPES = ".txt,.md,.pdf";

function inferSourceType(name: string, mime: string): "pdf" | "text" | "markdown" {
  if (mime === "application/pdf" || name.endsWith(".pdf")) return "pdf";
  if (name.endsWith(".md") || name.endsWith(".markdown")) return "markdown";
  return "text";
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function KnowledgeBase() {
  const [projects, setProjects] = useState<KBProject[]>([]);
  const [folders, setFolders] = useState<KBFolder[]>([]);
  const [docs, setDocs] = useState<KBItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<Map<string, string>>(new Map());
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<string | null>(null);
  const [uploadFolderId, setUploadFolderId] = useState<string | null>(null);
  const [uploadProjectId, setUploadProjectId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const addMenuRef = useRef<HTMLDivElement>(null);

  const { openPicker, downloadFile, loading: pickerLoading, error: pickerError, isConfigured: driveConfigured } =
    useGoogleDrivePicker();

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(searchQuery), 150);
    return () => clearTimeout(t);
  }, [searchQuery]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (addMenuRef.current && !addMenuRef.current.contains(e.target as Node)) {
        setShowAddMenu(false);
      }
    }
    if (showAddMenu) {
      document.addEventListener("mousedown", handleClick);
      return () => document.removeEventListener("mousedown", handleClick);
    }
  }, [showAddMenu]);

  useEffect(() => {
    if (!showAddMenu) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setShowAddMenu(false);
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [showAddMenu]);

  // ─── Data Fetching ──────────────────────────────────────

  const fetchAll = useCallback(async (opts?: { initProject?: boolean }) => {
    try {
      const [kbRes, foldersRes, projRes] = await Promise.all([
        fetch("/api/kb"),
        fetch("/api/kb/folders"),
        fetch("/api/kb/projects"),
      ]);
      if (!kbRes.ok || !foldersRes.ok || !projRes.ok) throw new Error("API error");
      const kbJson = await kbRes.json();
      const foldersJson = await foldersRes.json();
      const projJson = await projRes.json();

      setDocs(kbJson.items ?? []);
      setFolders(foldersJson.folders ?? []);
      setProjects(projJson.projects ?? []);

      if (opts?.initProject && (projJson.projects ?? []).length > 0) {
        setUploadProjectId(projJson.projects[0].id);
      }
    } catch {
      setError("Failed to load documents.");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAll({ initProject: true });
  }, [fetchAll]);

  useEffect(() => {
    const BASE_INTERVAL = 15_000;
    const MAX_INTERVAL = 60_000;
    let currentInterval = BASE_INTERVAL;
    let timerId: ReturnType<typeof setTimeout>;
    let unmounted = false;

    async function poll() {
      try {
        const [kbRes, foldersRes] = await Promise.all([
          fetch("/api/kb"),
          fetch("/api/kb/folders"),
        ]);
        if (unmounted) return; // don't schedule after unmount
        if (!kbRes.ok || !foldersRes.ok) {
          currentInterval = Math.min(currentInterval * 2, MAX_INTERVAL);
        } else {
          const kbJson = await kbRes.json();
          const foldersJson = await foldersRes.json();
          if (kbJson.items) setDocs(kbJson.items);
          if (foldersJson.folders) setFolders(foldersJson.folders);
          currentInterval = BASE_INTERVAL;
        }
      } catch {
        if (unmounted) return;
        currentInterval = Math.min(currentInterval * 2, MAX_INTERVAL);
      }
      if (!unmounted) {
        timerId = setTimeout(poll, currentInterval);
      }
    }

    timerId = setTimeout(poll, currentInterval);
    return () => {
      unmounted = true;
      clearTimeout(timerId);
    };
  }, []);

  // ─── Tree ───────────────────────────────────────────────

  const tree = useMemo(() => buildTree(projects, folders, docs), [projects, folders, docs]);

  const displayTree = useMemo(
    () => (debouncedQuery ? filterTree(tree, debouncedQuery) : tree),
    [tree, debouncedQuery],
  );

  const selectedDoc = useMemo(
    () => docs.find((d) => d.id === selectedDocId) ?? null,
    [docs, selectedDocId],
  );

  const breadcrumb = useMemo(
    () => (selectedDocId ? buildBreadcrumb(selectedDocId, tree) : []),
    [selectedDocId, tree],
  );

  // ─── Folder CRUD ────────────────────────────────────────

  async function handleCreateFolder(name: string, projectId: string, parentId: string | null) {
    try {
      const res = await fetch("/api/kb/folders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, project_id: projectId, parent_id: parentId }),
      });
      if (!res.ok) throw new Error("Failed to create folder");
      const { folder } = await res.json();
      setFolders((prev) => [...prev, folder]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create folder");
    }
  }

  async function handleRenameFolder(id: string, name: string) {
    try {
      const res = await fetch(`/api/kb/folders/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error("Failed to rename folder");
      const { folder } = await res.json();
      setFolders((prev) => prev.map((f) => (f.id === id ? folder : f)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rename folder");
    }
  }

  async function handleDeleteFolder(id: string) {
    if (!confirm("Delete this folder and all its contents?")) return;
    try {
      const res = await fetch(`/api/kb/folders/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete folder");
      await fetchAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete folder");
    }
  }

  // ─── Node Selection ─────────────────────────────────────

  function handleNodeSelect(node: TreeNode) {
    if (node.type === "doc") {
      setSelectedDocId(node.id);
    }
  }

  // ─── Upload Logic ───────────────────────────────────────

  async function uploadBlob(
    blob: Blob,
    fileName: string,
    mimeType: string,
    driveFileId: string,
    sourceType: string,
    progressMap: Map<string, string>,
    progressKey: string,
  ) {
    progressMap.set(progressKey, "Uploading...");
    setUploadProgress(new Map(progressMap));

    const fileId = crypto.randomUUID();
    const storagePath = `uploads/${fileId}/${fileName}`;

    const formData = new FormData();
    formData.append("file", blob, fileName);
    formData.append("storage_path", storagePath);
    formData.append("content_type", blob.type || mimeType || "application/octet-stream");
    const uploadRes = await fetch("/api/kb/files", { method: "POST", body: formData });
    if (!uploadRes.ok) {
      const { error } = await uploadRes.json().catch(() => ({ error: "Upload failed" }));
      throw new Error(error);
    }

    progressMap.set(progressKey, "Queuing...");
    setUploadProgress(new Map(progressMap));

    const queueBody: Record<string, unknown> = {
      action: "queue",
      file_name: fileName,
      drive_file_id: driveFileId,
      mime_type: mimeType,
      storage_path: storagePath,
      file_size_bytes: blob.size,
      source_type: sourceType,
      title: fileName.replace(/\.[^.]+$/, ""),
    };
    if (uploadProjectId) queueBody.project_id = uploadProjectId;
    if (uploadFolderId) queueBody.folder_id = uploadFolderId;

    const queueRes = await fetch("/api/kb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(queueBody),
    });
    if (!queueRes.ok) {
      const { error } = await queueRes.json().catch(() => ({ error: "Queue failed" }));
      throw new Error(error);
    }

    progressMap.set(progressKey, "Done");
    setUploadProgress(new Map(progressMap));
  }

  async function handleAddFromDrive() {
    setShowAddMenu(false);
    setError(null);
    setUploading(true);
    try {
      const files = await openPicker();
      if (!files.length) { setUploading(false); return; }

      const progress = new Map<string, string>();
      for (const file of files) {
        progress.set(file.id, "Downloading...");
        setUploadProgress(new Map(progress));
        try {
          if (file.sizeBytes > MAX_FILE_SIZE) {
            progress.set(file.id, `Too large (${formatSize(file.sizeBytes)})`);
            setUploadProgress(new Map(progress));
            continue;
          }
          const dupRes = await fetch("/api/kb", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "check_duplicate", drive_file_id: file.id }),
          });
          if (dupRes.ok && (await dupRes.json()).exists) {
            progress.set(file.id, "Already queued");
            setUploadProgress(new Map(progress));
            continue;
          }
          const { blob, sourceType } = await downloadFile(file);
          await uploadBlob(blob, file.name, file.mimeType, file.id, sourceType, progress, file.id);
        } catch (err) {
          progress.set(file.id, `Error: ${err instanceof Error ? err.message : "Upload failed"}`);
          setUploadProgress(new Map(progress));
        }
      }
      setTimeout(() => { setUploadProgress(new Map()); fetchAll(); }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to open picker");
    } finally {
      setUploading(false);
    }
  }

  async function handleAddFromComputer() {
    setShowAddMenu(false);
    fileInputRef.current?.click();
  }

  async function handleLocalFileSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;
    setError(null);
    setUploading(true);
    const progress = new Map<string, string>();
    try {
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        const progressKey = file.name;
        progress.set(progressKey, "Reading...");
        setUploadProgress(new Map(progress));
        try {
          if (file.size > MAX_FILE_SIZE) {
            progress.set(progressKey, `Too large (${formatSize(file.size)})`);
            setUploadProgress(new Map(progress));
            continue;
          }
          const sourceType = inferSourceType(file.name, file.type);
          const blob = new Blob([await file.arrayBuffer()], { type: file.type });
          await uploadBlob(blob, file.name, file.type, `local-${crypto.randomUUID()}`, sourceType, progress, progressKey);
        } catch (err) {
          progress.set(progressKey, `Error: ${err instanceof Error ? err.message : "Upload failed"}`);
          setUploadProgress(new Map(progress));
        }
      }
    } finally {
      setTimeout(() => { setUploadProgress(new Map()); fetchAll(); }, 2000);
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDeleteDoc() {
    if (!selectedDoc) return;
    if (!confirm(`Delete "${selectedDoc.title || selectedDoc.file_name}"?`)) return;
    try {
      const res = await fetch("/api/kb", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: selectedDoc.id }),
      });
      if (!res.ok) throw new Error("Delete failed");
      setDocs((prev) => prev.filter((d) => d.id !== selectedDoc.id));
      setSelectedDocId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  // ─── Sync Repo Docs ────────────────────────────────────

  async function handleSyncDocs() {
    setSyncing(true);
    setSyncResult(null);
    setError(null);
    try {
      const res = await fetch("/api/admin/sync-docs", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Sync failed");
      const errCount = data.errors?.length ?? 0;
      const msg = `Synced ${data.synced}, skipped ${data.skipped} of ${data.total}`;
      setSyncResult(errCount > 0 ? `${msg} (${errCount} errors)` : msg);
      if (errCount > 0) setError(data.errors[0]);
      fetchAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sync failed");
    } finally {
      setSyncing(false);
      setTimeout(() => setSyncResult(null), 6000);
    }
  }

  // ─── Loading ────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3" role="status">
        <div className="w-5 h-5 border-2 border-ink-faint border-t-ink-tertiary rounded-full animate-spin" />
        <span className="text-ink-tertiary text-[13px]">Loading...</span>
      </div>
    );
  }

  // ─── Render ─────────────────────────────────────────────

  return (
    <div className="flex h-[calc(100dvh-112px)]">
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_FILE_TYPES}
        multiple
        className="hidden"
        onChange={handleLocalFileSelected}
      />

      {/* ── Sidebar ──────────────────────────────────────── */}
      <div
        className={`w-full md:w-[280px] md:shrink-0 md:border-r md:border-edge-subtle flex flex-col bg-canvas ${
          selectedDocId ? "hidden md:flex" : "flex"
        }`}
      >
        <div className="shrink-0 px-3 pt-4 pb-2 space-y-2">
          {/* Search */}
          <div className="relative">
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-faint pointer-events-none"
              fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
            </svg>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search docs..."
              className="w-full pl-9 pr-8 py-2 text-[13px] font-mono bg-canvas border border-edge-subtle rounded-lg text-ink placeholder:text-ink-faint focus:border-gold/30 focus:outline-none focus:ring-1 focus:ring-gold/20 transition-colors"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink-tertiary transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>

          {/* Add file dropdown + Sync */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative" ref={addMenuRef}>
                <button
                  onClick={() => setShowAddMenu(!showAddMenu)}
                  disabled={uploading || pickerLoading}
                  className="flex items-center gap-1.5 text-[12px] font-medium text-gold hover:text-gold-dim transition-colors touch-manipulation disabled:opacity-50"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                  </svg>
                  {uploading ? "Uploading..." : "Add file"}
                  <svg className="w-3 h-3 opacity-50" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
                  </svg>
                </button>

              {showAddMenu && (
                <div className="absolute top-full left-0 mt-1 w-56 bg-surface-elevated border border-edge-subtle rounded-lg shadow-lg z-20 py-1">
                  <div className="px-3 py-2 border-b border-edge-subtle space-y-1.5">
                    <div>
                      <label className="text-[10px] font-medium text-ink-faint uppercase tracking-wider block mb-1">Project</label>
                      <select
                        value={uploadProjectId || ""}
                        onChange={(e) => { setUploadProjectId(e.target.value || null); setUploadFolderId(null); }}
                        className="w-full text-[12px] bg-canvas border border-edge-subtle rounded-md px-2 py-1 text-ink focus:border-gold/30 focus:outline-none"
                      >
                        <option value="">No project</option>
                        {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                      </select>
                    </div>
                    {uploadProjectId && (
                      <div>
                        <label className="text-[10px] font-medium text-ink-faint uppercase tracking-wider block mb-1">Folder</label>
                        <select
                          value={uploadFolderId || ""}
                          onChange={(e) => setUploadFolderId(e.target.value || null)}
                          className="w-full text-[12px] bg-canvas border border-edge-subtle rounded-md px-2 py-1 text-ink focus:border-gold/30 focus:outline-none"
                        >
                          <option value="">Root (unfiled)</option>
                          {folders
                            .filter((f) => f.project_id === uploadProjectId)
                            .map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
                        </select>
                      </div>
                    )}
                  </div>

                  {driveConfigured && (
                  <button
                    onClick={handleAddFromDrive}
                    className="w-full text-left px-3 py-2 text-[12px] text-ink-secondary hover:bg-surface-hover transition-colors flex items-center gap-2"
                  >
                    <svg className="w-4 h-4 text-ink-faint" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M7.71 3.5L1.15 15l4.58 7.5h13.54l4.58-7.5L17.29 3.5H7.71zm.58 1h8.42l5.5 10h-4.46l-5-8.5-.75 1.28L16.24 14.5H3.76l5.5-10h-.97zm-.83 11h9.08l-3.79 6.5H8.25L4.46 15.5z" />
                    </svg>
                    From Google Drive
                  </button>
                  )}
                  <button
                    onClick={handleAddFromComputer}
                    className="w-full text-left px-3 py-2 text-[12px] text-ink-secondary hover:bg-surface-hover transition-colors flex items-center gap-2"
                  >
                    <svg className="w-4 h-4 text-ink-faint" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                    </svg>
                    From computer
                    <span className="text-[10px] text-ink-faint ml-auto">.txt .md .pdf</span>
                  </button>
                </div>
              )}
            </div>

              <button
                onClick={handleSyncDocs}
                disabled={syncing}
                className="flex items-center gap-1.5 text-[12px] font-medium text-ink-tertiary hover:text-ink-secondary transition-colors touch-manipulation disabled:opacity-50"
              >
                <svg className={`w-3.5 h-3.5 ${syncing ? "animate-spin" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182M2.985 19.644l3.181-3.182" />
                </svg>
                {syncing ? "Syncing..." : "Sync repo docs"}
              </button>
            </div>

            {syncResult && (
              <span className="text-[11px] text-type-lesson font-medium">{syncResult}</span>
            )}
          </div>
        </div>

        {/* Upload progress */}
        {uploadProgress.size > 0 && (
          <div className="shrink-0 mx-3 mb-2 space-y-1">
            {Array.from(uploadProgress.entries()).map(([id, status]) => (
              <div key={id} className="flex items-center justify-between px-2.5 py-1.5 bg-surface rounded-md text-[11px]">
                <span className="text-ink-secondary truncate mr-2">{id.length > 20 ? id.slice(0, 20) + "..." : id}</span>
                <span className={`shrink-0 font-medium ${status.startsWith("Error") ? "text-type-error" : status === "Done" ? "text-type-lesson" : "text-ink-tertiary"}`}>
                  {status}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Error */}
        {(error || pickerError) && (
          <div className="shrink-0 mx-3 mb-2 p-2.5 bg-type-error/10 border border-type-error/20 rounded-lg text-[12px] text-type-error" role="alert">
            {error || pickerError}
          </div>
        )}

        {/* Folder Tree */}
        <FolderTree
          tree={displayTree}
          selectedId={selectedDocId}
          onSelect={handleNodeSelect}
          onCreateFolder={handleCreateFolder}
          onRenameFolder={handleRenameFolder}
          onDeleteFolder={handleDeleteFolder}
        />
      </div>

      {/* ── Detail Pane ──────────────────────────────────── */}
      <div className={`flex-1 min-w-0 overflow-y-auto ${selectedDocId ? "flex flex-col" : "hidden md:flex md:flex-col"}`}>
        {selectedDoc ? (
          <DocPreview
            doc={selectedDoc}
            breadcrumb={breadcrumb}
            onBack={() => setSelectedDocId(null)}
            onDelete={handleDeleteDoc}
          />
        ) : (
          <div className="hidden md:flex flex-1 flex-col items-center justify-center text-ink-faint">
            <svg className="w-10 h-10 mb-3 opacity-40" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 0 1 4.5 9.75h15A2.25 2.25 0 0 1 21.75 12v.75m-8.69-6.44-2.12-2.12a1.5 1.5 0 0 0-1.061-.44H4.5A2.25 2.25 0 0 0 2.25 6v12a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9a2.25 2.25 0 0 0-2.25-2.25h-5.379a1.5 1.5 0 0 1-1.06-.44Z" />
            </svg>
            <span className="text-[13px]">Select a document</span>
          </div>
        )}
      </div>
    </div>
  );
}
