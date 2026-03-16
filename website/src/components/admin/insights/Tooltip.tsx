import React, { useRef, useState } from "react";

export default function Tooltip({
  children,
  content,
}: {
  children: React.ReactNode;
  content: React.ReactNode;
}) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const ref = useRef<HTMLDivElement>(null);

  function handleEnter() {
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setPos({ x: rect.left + rect.width / 2, y: rect.top });
      setShow(true);
    }
  }

  return (
    <div ref={ref} onMouseEnter={handleEnter} onMouseLeave={() => setShow(false)} onFocus={handleEnter} onBlur={() => setShow(false)}>
      {children}
      {show && (
        <div
          className="fixed z-50 pointer-events-none"
          role="tooltip"
          style={{ left: pos.x, top: pos.y, transform: 'translate(-50%, -100%) translateY(-8px)' }}
        >
          <div className="bg-surface-elevated border border-edge-strong rounded-lg shadow-lg px-3 py-2 text-[16px] text-ink-secondary whitespace-nowrap">
            {content}
          </div>
        </div>
      )}
    </div>
  );
}
