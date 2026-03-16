import { useCallback, useEffect, useState } from "react";

interface UseKeyboardNavOptions {
  /** Data attribute name used to find elements (e.g., "data-job-id") */
  dataAttribute: string;
  /** All visible IDs in order */
  ids: string[];
  /** Called when Enter/e is pressed on a focused item */
  onSelect?: (id: string) => void;
  /** Extra key handlers: key -> handler(focusedId) */
  extraKeys?: Record<string, (id: string) => void>;
  /** Whether the hook is active (default true) */
  enabled?: boolean;
}

export function useKeyboardNav({
  dataAttribute,
  ids,
  onSelect,
  extraKeys,
  enabled = true,
}: UseKeyboardNavOptions) {
  const [focusedIndex, setFocusedIndex] = useState(-1);

  // Reset focus when ids change
  useEffect(() => {
    setFocusedIndex(-1);
  }, [ids]);

  useEffect(() => {
    if (!enabled) return;

    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      switch (e.key) {
        case "j": {
          e.preventDefault();
          setFocusedIndex((prev) => {
            const next = Math.min(prev + 1, ids.length - 1);
            const el = document.querySelector(`[${dataAttribute}="${ids[next]}"]`);
            el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
            return next;
          });
          break;
        }
        case "k": {
          e.preventDefault();
          setFocusedIndex((prev) => {
            const next = Math.max(prev - 1, 0);
            const el = document.querySelector(`[${dataAttribute}="${ids[next]}"]`);
            el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
            return next;
          });
          break;
        }
        case "Enter":
        case "e": {
          if (focusedIndex >= 0 && focusedIndex < ids.length) {
            e.preventDefault();
            onSelect?.(ids[focusedIndex]);
          }
          break;
        }
        case "Escape": {
          if (focusedIndex >= 0) {
            e.preventDefault();
            setFocusedIndex(-1);
          }
          break;
        }
        default: {
          if (extraKeys && e.key in extraKeys && focusedIndex >= 0 && focusedIndex < ids.length) {
            e.preventDefault();
            extraKeys[e.key](ids[focusedIndex]);
          }
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [enabled, ids, focusedIndex, onSelect, extraKeys, dataAttribute]);

  return { focusedIndex, focusedId: focusedIndex >= 0 ? ids[focusedIndex] : null, setFocusedIndex };
}
