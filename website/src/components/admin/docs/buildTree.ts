import type { KBProject, KBFolder, KBItem, TreeNode } from "./types";

export function buildTree(
  projects: KBProject[],
  folders: KBFolder[],
  docs: KBItem[],
): TreeNode[] {
  const foldersByParent = new Map<string | null, KBFolder[]>();
  for (const f of folders) {
    const key = f.parent_id;
    const arr = foldersByParent.get(key) ?? [];
    arr.push(f);
    foldersByParent.set(key, arr);
  }

  const docsByFolder = new Map<string | null, KBItem[]>();
  for (const d of docs) {
    const key = d.folder_id ?? null;
    const arr = docsByFolder.get(key) ?? [];
    arr.push(d);
    docsByFolder.set(key, arr);
  }

  function buildFolderNode(folder: KBFolder): TreeNode {
    const childFolders = (foldersByParent.get(folder.id) ?? [])
      .sort((a, b) => a.sort_order - b.sort_order || a.name.localeCompare(b.name));
    const childDocs = (docsByFolder.get(folder.id) ?? [])
      .sort((a, b) => (a.title ?? a.file_name).localeCompare(b.title ?? b.file_name));

    return {
      type: "folder",
      id: folder.id,
      name: folder.name,
      projectId: folder.project_id,
      parentFolderId: folder.parent_id,
      children: [
        ...childFolders.map(buildFolderNode),
        ...childDocs.map(docToNode),
      ],
    };
  }

  function docToNode(doc: KBItem): TreeNode {
    return {
      type: "doc",
      id: doc.id,
      name: doc.title ?? doc.file_name,
      projectId: doc.project_id ?? "",
      parentFolderId: doc.folder_id,
      children: [],
      data: doc,
    };
  }

  const projectIds = new Set(projects.map((p) => p.id));

  const projectNodes: TreeNode[] = projects.map((project) => {
    const rootFolders = (foldersByParent.get(null) ?? [])
      .filter((f) => f.project_id === project.id)
      .sort((a, b) => a.sort_order - b.sort_order || a.name.localeCompare(b.name));

    const unfiledDocs = (docsByFolder.get(null) ?? [])
      .filter((d) => d.project_id === project.id)
      .sort((a, b) => (a.title ?? a.file_name).localeCompare(b.title ?? b.file_name));

    return {
      type: "project" as const,
      id: project.id,
      name: project.name,
      projectId: project.id,
      parentFolderId: null,
      color: project.color,
      children: [
        ...rootFolders.map(buildFolderNode),
        ...unfiledDocs.map(docToNode),
      ],
    };
  });

  // Collect docs with no project or a project_id that doesn't match any project
  const orphanedDocs = (docsByFolder.get(null) ?? [])
    .filter((d) => !d.project_id || !projectIds.has(d.project_id))
    .sort((a, b) => (a.title ?? a.file_name).localeCompare(b.title ?? b.file_name));

  if (orphanedDocs.length > 0) {
    projectNodes.push({
      type: "project",
      id: "__unfiled__",
      name: "Unfiled",
      projectId: "__unfiled__",
      parentFolderId: null,
      color: "#6b7280",
      children: orphanedDocs.map(docToNode),
    });
  }

  return projectNodes;
}

export function filterTree(nodes: TreeNode[], query: string): TreeNode[] {
  const q = query.toLowerCase();

  function filterNode(node: TreeNode): TreeNode | null {
    if (node.type === "doc") {
      return node.name.toLowerCase().includes(q) ? node : null;
    }

    const filteredChildren = node.children
      .map(filterNode)
      .filter((n): n is TreeNode => n !== null);

    if (filteredChildren.length > 0 || node.name.toLowerCase().includes(q)) {
      return { ...node, children: filteredChildren };
    }

    return null;
  }

  return nodes.map(filterNode).filter((n): n is TreeNode => n !== null);
}

export function buildBreadcrumb(
  nodeId: string,
  tree: TreeNode[],
): { id: string; name: string; type: "project" | "folder" | "doc" }[] {
  const path: { id: string; name: string; type: "project" | "folder" | "doc" }[] = [];

  function search(nodes: TreeNode[], ancestors: typeof path): boolean {
    for (const node of nodes) {
      const current = [...ancestors, { id: node.id, name: node.name, type: node.type }];
      if (node.id === nodeId) {
        path.push(...current);
        return true;
      }
      if (search(node.children, current)) return true;
    }
    return false;
  }

  search(tree, []);
  return path;
}
