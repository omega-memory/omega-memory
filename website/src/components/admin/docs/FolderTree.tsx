import { useCallback, useEffect, useRef, useState } from "react";
import type { TreeNode } from "./types";

interface FolderTreeProps {
  tree: TreeNode[];
  selectedId: string | null;
  onSelect: (node: TreeNode) => void;
  onCreateFolder: (name: string, projectId: string, parentId: string | null) => void;
  onRenameFolder: (id: string, name: string) => void;
  onDeleteFolder: (id: string) => void;
}

export default function FolderTree({
  tree,
  selectedId,
  onSelect,
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
}: FolderTreeProps) {
  const [expanded, setExpanded] = useState<Set<string>>(() => {
    return new Set(tree.map((n) => n.id));
  });
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    node: TreeNode;
  } | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [newFolderParent, setNewFolderParent] = useState<{
    projectId: string;
    parentId: string | null;
  } | null>(null);

  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent, node: TreeNode) => {
      if (node.type === "doc") return;
      e.preventDefault();
      setContextMenu({ x: e.clientX, y: e.clientY, node });
    },
    [],
  );

  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Dismiss context menu on scroll (KB-8)
  useEffect(() => {
    if (!contextMenu) return;

    const handleScroll = () => setContextMenu(null);
    const container = scrollContainerRef.current;

    // Listen on both the scrollable container and window for scroll events
    container?.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      container?.removeEventListener("scroll", handleScroll);
      window.removeEventListener("scroll", handleScroll);
    };
  }, [contextMenu]);

  return (
    <div ref={scrollContainerRef} className="flex-1 overflow-y-auto min-h-0" onClick={closeContextMenu}>
      <div className="py-1">
        {tree.map((node) => (
          <TreeNodeRow
            key={node.id}
            node={node}
            depth={0}
            expanded={expanded}
            selectedId={selectedId}
            renamingId={renamingId}
            onToggle={toggle}
            onSelect={onSelect}
            onContextMenu={handleContextMenu}
            onRenameSubmit={(id, name) => {
              onRenameFolder(id, name);
              setRenamingId(null);
            }}
            onRenameCancel={() => setRenamingId(null)}
            newFolderParent={newFolderParent}
            onNewFolderSubmit={(name, projectId, parentId) => {
              onCreateFolder(name, projectId, parentId);
              setNewFolderParent(null);
            }}
            onNewFolderCancel={() => setNewFolderParent(null)}
          />
        ))}
      </div>

      {contextMenu && (
        <div
          className="fixed z-50 bg-surface-elevated border border-edge-subtle rounded-lg shadow-lg py-1 min-w-[160px]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            className="w-full text-left px-3 py-1.5 text-[12px] text-ink-secondary hover:bg-surface-hover transition-colors"
            onClick={() => {
              setNewFolderParent({
                projectId: contextMenu.node.projectId,
                parentId:
                  contextMenu.node.type === "project"
                    ? null
                    : contextMenu.node.id,
              });
              setExpanded((prev) => new Set(prev).add(contextMenu.node.id));
              closeContextMenu();
            }}
          >
            New folder
          </button>
          {contextMenu.node.type === "folder" && (
            <>
              <button
                className="w-full text-left px-3 py-1.5 text-[12px] text-ink-secondary hover:bg-surface-hover transition-colors"
                onClick={() => {
                  setRenamingId(contextMenu.node.id);
                  closeContextMenu();
                }}
              >
                Rename
              </button>
              <button
                className="w-full text-left px-3 py-1.5 text-[12px] text-type-error hover:bg-type-error/10 transition-colors"
                onClick={() => {
                  onDeleteFolder(contextMenu.node.id);
                  closeContextMenu();
                }}
              >
                Delete
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

interface TreeNodeRowProps {
  node: TreeNode;
  depth: number;
  expanded: Set<string>;
  selectedId: string | null;
  renamingId: string | null;
  onToggle: (id: string) => void;
  onSelect: (node: TreeNode) => void;
  onContextMenu: (e: React.MouseEvent, node: TreeNode) => void;
  onRenameSubmit: (id: string, name: string) => void;
  onRenameCancel: () => void;
  newFolderParent: { projectId: string; parentId: string | null } | null;
  onNewFolderSubmit: (name: string, projectId: string, parentId: string | null) => void;
  onNewFolderCancel: () => void;
}

function TreeNodeRow({
  node,
  depth,
  expanded,
  selectedId,
  renamingId,
  onToggle,
  onSelect,
  onContextMenu,
  onRenameSubmit,
  onRenameCancel,
  newFolderParent,
  onNewFolderSubmit,
  onNewFolderCancel,
}: TreeNodeRowProps) {
  const isExpanded = expanded.has(node.id);
  const isSelected = selectedId === node.id;
  const isRenaming = renamingId === node.id;
  const isExpandable = node.type !== "doc";

  const showNewFolderHere =
    newFolderParent &&
    ((node.type === "project" && newFolderParent.parentId === null && newFolderParent.projectId === node.id) ||
      (node.type === "folder" && newFolderParent.parentId === node.id));

  const paddingLeft = depth === 0 ? 8 : 8 + depth * 16;

  return (
    <>
      <div
        className={`flex items-center gap-1.5 py-1 px-2 rounded-md mx-1 cursor-pointer transition-colors text-[13px] ${
          isSelected
            ? "bg-surface-elevated text-ink"
            : "text-ink-secondary hover:bg-surface-hover"
        }`}
        style={{ paddingLeft }}
        onClick={() => {
          if (isExpandable) onToggle(node.id);
          onSelect(node);
        }}
        onContextMenu={(e) => onContextMenu(e, node)}
      >
        {isExpandable ? (
          <svg
            className={`w-3.5 h-3.5 shrink-0 text-ink-faint transition-transform ${
              isExpanded ? "rotate-90" : ""
            }`}
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
          </svg>
        ) : (
          <span className="w-3.5 shrink-0" />
        )}

        {node.type === "project" ? (
          <span
            className="w-2.5 h-2.5 rounded-full shrink-0"
            style={{ backgroundColor: node.color ?? "#888" }}
          />
        ) : node.type === "folder" ? (
          <svg
            className="w-4 h-4 shrink-0 text-gold/70"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M2.25 12.75V12A2.25 2.25 0 0 1 4.5 9.75h15A2.25 2.25 0 0 1 21.75 12v.75m-8.69-6.44-2.12-2.12a1.5 1.5 0 0 0-1.061-.44H4.5A2.25 2.25 0 0 0 2.25 6v12a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9a2.25 2.25 0 0 0-2.25-2.25h-5.379a1.5 1.5 0 0 1-1.06-.44Z"
            />
          </svg>
        ) : (
          <svg
            className="w-4 h-4 shrink-0 text-ink-faint"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
            />
          </svg>
        )}

        {isRenaming ? (
          <InlineInput
            defaultValue={node.name}
            onSubmit={(val) => onRenameSubmit(node.id, val)}
            onCancel={onRenameCancel}
          />
        ) : (
          <span className="truncate font-medium">{node.name}</span>
        )}

        {node.type === "doc" && node.data && node.data.status !== "synced" && (
          <span
            className={`w-1.5 h-1.5 rounded-full shrink-0 ml-auto ${
              node.data.status === "pending"
                ? "bg-amber-400"
                : node.data.status === "processing"
                  ? "bg-type-decision"
                  : "bg-type-error"
            }`}
            title={node.data.status}
          />
        )}
      </div>

      {isExpanded && (
        <>
          {node.children.map((child) => (
            <TreeNodeRow
              key={child.id}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              selectedId={selectedId}
              renamingId={renamingId}
              onToggle={onToggle}
              onSelect={onSelect}
              onContextMenu={onContextMenu}
              onRenameSubmit={onRenameSubmit}
              onRenameCancel={onRenameCancel}
              newFolderParent={newFolderParent}
              onNewFolderSubmit={onNewFolderSubmit}
              onNewFolderCancel={onNewFolderCancel}
            />
          ))}
          {showNewFolderHere && (
            <div
              className="flex items-center gap-1.5 py-1 px-2 mx-1"
              style={{ paddingLeft: paddingLeft + 16 }}
            >
              <svg
                className="w-4 h-4 shrink-0 text-gold/70"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 10.5v6m3-3H9m4.06-7.19-2.12-2.12a1.5 1.5 0 0 0-1.061-.44H4.5A2.25 2.25 0 0 0 2.25 6v12a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9a2.25 2.25 0 0 0-2.25-2.25h-5.379a1.5 1.5 0 0 1-1.06-.44Z"
                />
              </svg>
              <InlineInput
                defaultValue=""
                placeholder="Folder name..."
                onSubmit={(val) =>
                  onNewFolderSubmit(
                    val,
                    newFolderParent.projectId,
                    newFolderParent.parentId,
                  )
                }
                onCancel={onNewFolderCancel}
              />
            </div>
          )}
        </>
      )}
    </>
  );
}

function InlineInput({
  defaultValue,
  placeholder,
  onSubmit,
  onCancel,
}: {
  defaultValue: string;
  placeholder?: string;
  onSubmit: (value: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(defaultValue);

  const setRef = useCallback(
    (el: HTMLInputElement | null) => {
      if (el) {
        el.focus();
        if (defaultValue) el.select();
      }
    },
    [defaultValue],
  );

  return (
    <input
      ref={setRef}
      type="text"
      value={value}
      placeholder={placeholder}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter" && value.trim()) {
          onSubmit(value.trim());
        } else if (e.key === "Escape") {
          onCancel();
        }
      }}
      onBlur={() => {
        if (value.trim() && value.trim() !== defaultValue) {
          onSubmit(value.trim());
        } else {
          onCancel();
        }
      }}
      className="flex-1 min-w-0 text-[13px] font-medium bg-transparent border border-gold/30 rounded px-1.5 py-0.5 text-ink focus:outline-none focus:ring-1 focus:ring-gold/20"
    />
  );
}
