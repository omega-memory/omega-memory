export interface KBProject {
  id: string;
  name: string;
  slug: string;
  color: string;
  created_at: string;
}

export interface KBFolder {
  id: string;
  name: string;
  parent_id: string | null;
  project_id: string;
  sort_order: number;
  created_at: string;
}

export interface KBItem {
  id: string;
  file_name: string;
  drive_file_id: string;
  mime_type: string;
  source_type: string;
  storage_path: string;
  title: string | null;
  status: "pending" | "processing" | "synced" | "error";
  error_message: string | null;
  file_size_bytes: number;
  created_at: string;
  synced_at: string | null;
  project_id: string | null;
  folder_id: string | null;
}

export interface TreeNode {
  type: "project" | "folder" | "doc";
  id: string;
  name: string;
  projectId: string;
  parentFolderId: string | null;
  children: TreeNode[];
  data?: KBItem;
  color?: string;
}
