import { useEffect, useRef } from "react";

interface AmbientEvent {
  type: "agent_joined" | "agent_left" | "file_conflict" | "memory_spike";
  message: string;
  timestamp: number;
}

const ICONS: Record<AmbientEvent["type"], { svg: string; color: string }> = {
  agent_joined: {
    svg: "M18 7.5v3m0 0v3m0-3h3m-3 0h-3m-2.25-4.125a3.375 3.375 0 1 1-6.75 0 3.375 3.375 0 0 1 6.75 0ZM3 19.235v-.11a6.375 6.375 0 0 1 12.75 0v.109A12.318 12.318 0 0 1 9.374 21c-2.331 0-4.512-.645-6.374-1.766Z",
    color: "text-emerald-400",
  },
  agent_left: {
    svg: "M22 10.5h-6m-2.25-4.125a3.375 3.375 0 1 1-6.75 0 3.375 3.375 0 0 1 6.75 0ZM4 19.235v-.11a6.375 6.375 0 0 1 12.75 0v.109A12.318 12.318 0 0 1 10.374 21c-2.331 0-4.512-.645-6.374-1.766Z",
    color: "text-ink-tertiary",
  },
  file_conflict: {
    svg: "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z",
    color: "text-type-error",
  },
  memory_spike: {
    svg: "M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z",
    color: "text-gold",
  },
};

const AUTO_DISMISS_MS = 6000;

export default function AmbientToast({
  events,
  onDismiss,
}: {
  events: AmbientEvent[];
  onDismiss: (timestamp: number) => void;
}) {
  const timersRef = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  useEffect(() => {
    // Register new timers for events that don't have one
    for (const event of events) {
      if (!timersRef.current.has(event.timestamp)) {
        const timer = setTimeout(() => {
          onDismiss(event.timestamp);
          timersRef.current.delete(event.timestamp);
        }, AUTO_DISMISS_MS);
        timersRef.current.set(event.timestamp, timer);
      }
    }

    // Clean up timers for events no longer in the list
    const currentTimestamps = new Set(events.map((e) => e.timestamp));
    for (const [ts, timer] of timersRef.current.entries()) {
      if (!currentTimestamps.has(ts)) {
        clearTimeout(timer);
        timersRef.current.delete(ts);
      }
    }
  }, [events, onDismiss]);

  useEffect(() => {
    return () => {
      for (const timer of timersRef.current.values()) clearTimeout(timer);
    };
  }, []);

  if (events.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 pointer-events-none">
      {events.map((event) => {
        const icon = ICONS[event.type];
        return (
          <div
            key={event.timestamp}
            className="pointer-events-auto flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl bg-surface-elevated/95 backdrop-blur-md border border-edge shadow-lg toast-enter"
          >
            <svg
              className={`w-4 h-4 shrink-0 ${icon.color}`}
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d={icon.svg} />
            </svg>
            <span className="text-[12px] text-ink-secondary">{event.message}</span>
            <button
              onClick={() => onDismiss(event.timestamp)}
              className="ml-1 shrink-0 w-4 h-4 flex items-center justify-center rounded text-ink-faint hover:text-ink-tertiary transition-colors"
            >
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        );
      })}
    </div>
  );
}
