import { useCallback, useState, useRef } from "react";

export function useBulkSelection(ids: string[]) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const lastSelectedRef = useRef<string | null>(null);

  const isSelected = useCallback((id: string) => selectedIds.has(id), [selectedIds]);

  const toggle = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      lastSelectedRef.current = id;
      return next;
    });
  }, []);

  const selectRange = useCallback((id: string) => {
    if (!lastSelectedRef.current) {
      toggle(id);
      return;
    }
    const startIdx = ids.indexOf(lastSelectedRef.current);
    const endIdx = ids.indexOf(id);
    if (startIdx === -1 || endIdx === -1) {
      toggle(id);
      return;
    }
    const [from, to] = startIdx < endIdx ? [startIdx, endIdx] : [endIdx, startIdx];
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (let i = from; i <= to; i++) {
        next.add(ids[i]);
      }
      return next;
    });
    lastSelectedRef.current = id;
  }, [ids, toggle]);

  const toggleAll = useCallback(() => {
    setSelectedIds((prev) => {
      if (prev.size === ids.length) return new Set();
      return new Set(ids);
    });
  }, [ids]);

  const clear = useCallback(() => {
    setSelectedIds(new Set());
    lastSelectedRef.current = null;
  }, []);

  const handleClick = useCallback((id: string, e: React.MouseEvent) => {
    if (e.shiftKey) {
      e.preventDefault();
      selectRange(id);
    } else if (e.metaKey || e.ctrlKey) {
      e.preventDefault();
      toggle(id);
    }
  }, [selectRange, toggle]);

  return {
    selectedIds,
    isSelected,
    toggle,
    toggleAll,
    selectRange,
    clear,
    count: selectedIds.size,
    handleClick,
  };
}
