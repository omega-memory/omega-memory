import React, { useRef, useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";

interface TooltipProps {
  children: React.ReactElement;
  content: React.ReactNode;
  side?: "top" | "bottom";
  align?: "center" | "start" | "end";
  delayMs?: number;
}

export default function Tooltip({
  children,
  content,
  side = "bottom",
  align = "center",
  delayMs = 200,
}: TooltipProps) {
  const triggerRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tooltipId = useRef(`tooltip-${Math.random().toString(36).slice(2, 8)}`).current;

  const updatePosition = useCallback(() => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();

    let x: number;
    if (align === "start") x = rect.left;
    else if (align === "end") x = rect.right;
    else x = rect.left + rect.width / 2;

    const y = side === "top" ? rect.top - 8 : rect.bottom + 8;
    setCoords({ x, y });
  }, [side, align]);

  const show = useCallback(() => {
    timerRef.current = setTimeout(() => {
      updatePosition();
      setVisible(true);
    }, delayMs);
  }, [delayMs, updatePosition]);

  const hide = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    setVisible(false);
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const translateX =
    align === "start" ? "0%" : align === "end" ? "-100%" : "-50%";
  const translateY = side === "top" ? "-100%" : "0%";
  const originY = side === "top" ? "bottom" : "top";

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        aria-describedby={visible ? tooltipId : undefined}
        className="inline-flex"
      >
        {children}
      </div>
      {typeof document !== "undefined" &&
        createPortal(
          <AnimatePresence>
            {visible && (
              <motion.div
                id={tooltipId}
                role="tooltip"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ duration: 0.15, ease: "easeOut" }}
                className="fixed z-[9999] pointer-events-none"
                style={{
                  left: coords.x,
                  top: coords.y,
                  transform: `translate(${translateX}, ${translateY})`,
                  transformOrigin: `${align === "start" ? "left" : align === "end" ? "right" : "center"} ${originY}`,
                }}
              >
                {/* Arrow */}
                <div
                  className={`absolute left-1/2 -translate-x-1/2 w-2 h-2 rotate-45 bg-surface-elevated border-edge-strong ${
                    side === "top"
                      ? "bottom-[-5px] border-b border-r"
                      : "top-[-5px] border-t border-l"
                  }`}
                  style={
                    align === "start"
                      ? { left: 16 }
                      : align === "end"
                        ? { left: "auto", right: 16 }
                        : {}
                  }
                />
                {/* Content */}
                <div className="bg-surface-elevated border border-edge-strong rounded-lg shadow-lg px-3 py-2.5 text-[14px] max-w-[280px]">
                  {content}
                </div>
              </motion.div>
            )}
          </AnimatePresence>,
          document.body,
        )}
    </>
  );
}
