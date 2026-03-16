import { useCallback, useEffect, useMemo, useState } from "react";
import type { Memory } from "./feedScoring";

interface UseFeedKeyboardOptions {
  important: Memory[];
  grouped: [string, Memory[]][];
  memories: Memory[];
  expandedId: string | null;
  setExpandedId: (id: string | null) => void;
  toggleExpand: (id: string) => void;
  handleDismiss: (m: Memory) => void;
  handleArchive: (m: Memory) => void;
  filter: string;
  sortBy: string;
}

export function useFeedKeyboard({
  important,
  grouped,
  memories,
  expandedId,
  setExpandedId,
  toggleExpand,
  handleDismiss,
  handleArchive,
  filter,
  sortBy,
}: UseFeedKeyboardOptions) {
  const [focusedIndex, setFocusedIndex] = useState(-1);

  const allVisibleIds = useMemo(() => {
    const ids: string[] = [];
    important.forEach((m) => ids.push(m.id));
    grouped.forEach(([, items]) => items.forEach((m) => ids.push(m.id)));
    return ids;
  }, [important, grouped]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      switch (e.key) {
        case "j": {
          e.preventDefault();
          setFocusedIndex((prev) => {
            const next = Math.min(prev + 1, allVisibleIds.length - 1);
            const el = document.querySelector(`[data-memory-id="${allVisibleIds[next]}"]`);
            el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
            return next;
          });
          break;
        }
        case "k": {
          e.preventDefault();
          setFocusedIndex((prev) => {
            const next = Math.max(prev - 1, 0);
            const el = document.querySelector(`[data-memory-id="${allVisibleIds[next]}"]`);
            el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
            return next;
          });
          break;
        }
        case "e":
        case "Enter": {
          if (focusedIndex >= 0 && focusedIndex < allVisibleIds.length) {
            e.preventDefault();
            toggleExpand(allVisibleIds[focusedIndex]);
          }
          break;
        }
        case "d": {
          if (focusedIndex >= 0 && focusedIndex < allVisibleIds.length) {
            const m = memories.find((mem) => mem.id === allVisibleIds[focusedIndex]);
            if (m?.event_type === "reminder" && m.metadata?.reminder_status !== "dismissed") {
              e.preventDefault();
              handleDismiss(m);
            }
          }
          break;
        }
        case "a": {
          if (focusedIndex >= 0 && focusedIndex < allVisibleIds.length) {
            const m = memories.find((mem) => mem.id === allVisibleIds[focusedIndex]);
            if (m && !m.metadata?.archived_at) {
              e.preventDefault();
              handleArchive(m);
            }
          }
          break;
        }
        case "Escape": {
          if (expandedId) {
            e.preventDefault();
            setExpandedId(null);
          } else {
            setFocusedIndex(-1);
          }
          break;
        }
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [allVisibleIds, focusedIndex, expandedId, memories, setExpandedId, toggleExpand, handleDismiss, handleArchive]);

  // Reset focus when filter/sort changes
  useEffect(() => {
    setFocusedIndex(-1);
  }, [filter, sortBy]);

  return { focusedIndex, allVisibleIds };
}
