import { useCallback } from "react";
import type { ReplayEvent } from "./replay-utils";
import { findEventIndex } from "./replay-utils";

const SPEEDS = [1, 2, 5, 10] as const;

interface ReplayControlsProps {
  timeline: ReplayEvent[];
  playbackTime: number | null;
  isPlaying: boolean;
  playbackSpeed: number;
  onSetPlaybackTime: (t: number) => void;
  onSetPlaying: (playing: boolean) => void;
  onSetSpeed: (speed: number) => void;
}

function formatClock(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function ReplayControls({
  timeline,
  playbackTime,
  isPlaying,
  playbackSpeed,
  onSetPlaybackTime,
  onSetPlaying,
  onSetSpeed,
}: ReplayControlsProps) {
  if (timeline.length === 0 || playbackTime == null) return null;

  const currentIdx = findEventIndex(timeline, playbackTime);
  const totalEvents = timeline.length;

  const goToPrev = useCallback(() => {
    const idx = currentIdx > 0 ? currentIdx - 1 : 0;
    onSetPlaybackTime(timeline[idx].timestamp);
    onSetPlaying(false);
  }, [currentIdx, timeline, onSetPlaybackTime, onSetPlaying]);

  const goToNext = useCallback(() => {
    const idx = currentIdx < totalEvents - 1 ? currentIdx + 1 : totalEvents - 1;
    onSetPlaybackTime(timeline[idx].timestamp);
    onSetPlaying(false);
  }, [currentIdx, totalEvents, timeline, onSetPlaybackTime, onSetPlaying]);

  const reset = useCallback(() => {
    onSetPlaybackTime(timeline[0].timestamp);
    onSetPlaying(false);
  }, [timeline, onSetPlaybackTime, onSetPlaying]);

  const cycleSpeed = useCallback(() => {
    const idx = SPEEDS.indexOf(playbackSpeed as typeof SPEEDS[number]);
    const next = SPEEDS[(idx + 1) % SPEEDS.length];
    onSetSpeed(next);
  }, [playbackSpeed, onSetSpeed]);

  const firstTs = timeline[0].timestamp;
  const lastTs = timeline[timeline.length - 1].timestamp;
  const span = lastTs - firstTs;
  const scrubPct = span > 0 ? ((playbackTime - firstTs) / span) * 100 : 0;

  return (
    <div className="flex items-center gap-2 px-4 py-1.5 bg-surface-elevated border-t border-b border-edge/40">
      {/* Prev */}
      <button
        onClick={goToPrev}
        disabled={currentIdx <= 0}
        className="w-7 h-7 flex items-center justify-center rounded-md text-ink-faint hover:text-ink-secondary hover:bg-surface-hover transition-colors disabled:opacity-30"
        title="Previous event (Left arrow)"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        </svg>
      </button>

      {/* Play/Pause */}
      <button
        onClick={() => onSetPlaying(!isPlaying)}
        className="w-7 h-7 flex items-center justify-center rounded-md text-ink-faint hover:text-ink-secondary hover:bg-surface-hover transition-colors"
        title={isPlaying ? "Pause (Space)" : "Play (Space)"}
      >
        {isPlaying ? (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" rx="1" />
            <rect x="14" y="4" width="4" height="16" rx="1" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5.14v14.72a1 1 0 001.5.86l11-7.36a1 1 0 000-1.72l-11-7.36A1 1 0 008 5.14z" />
          </svg>
        )}
      </button>

      {/* Next */}
      <button
        onClick={goToNext}
        disabled={currentIdx >= totalEvents - 1}
        className="w-7 h-7 flex items-center justify-center rounded-md text-ink-faint hover:text-ink-secondary hover:bg-surface-hover transition-colors disabled:opacity-30"
        title="Next event (Right arrow)"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        </svg>
      </button>

      {/* Speed */}
      <button
        onClick={cycleSpeed}
        className={`text-[11px] font-mono px-2 py-1 rounded-md border transition-colors ${
          playbackSpeed > 1
            ? "bg-gold/10 text-gold border-gold/20"
            : "bg-surface text-ink-faint border-edge/40 hover:text-ink-secondary"
        }`}
        title="Cycle speed ([ / ])"
      >
        {playbackSpeed}x
      </button>

      {/* Scrubber track */}
      <div className="flex-1 relative h-6 flex items-center group cursor-pointer mx-2">
        <div className="w-full h-1 rounded-full bg-edge/40 relative">
          {/* Filled portion */}
          <div
            className="absolute top-0 left-0 h-full rounded-full bg-gold/40"
            style={{ width: `${scrubPct}%` }}
          />
          {/* Thumb */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-gold shadow-md shadow-gold/30 group-hover:scale-125 transition-transform"
            style={{ left: `calc(${scrubPct}% - 6px)` }}
          />
        </div>
        {/* Invisible click target for scrubbing */}
        <input
          type="range"
          min={firstTs}
          max={lastTs}
          value={playbackTime}
          onChange={(e) => {
            onSetPlaybackTime(Number(e.target.value));
            onSetPlaying(false);
          }}
          className="absolute inset-0 w-full opacity-0 cursor-pointer"
          step={1}
        />
      </div>

      {/* Event counter */}
      <span className="text-[10px] font-mono text-ink-faint whitespace-nowrap">
        {currentIdx + 1}/{totalEvents}
      </span>

      {/* Clock */}
      <span className="text-[11px] font-mono text-ink-secondary whitespace-nowrap">
        {formatClock(playbackTime)}
      </span>

      {/* Reset */}
      <button
        onClick={reset}
        className="w-7 h-7 flex items-center justify-center rounded-md text-ink-faint hover:text-ink-secondary hover:bg-surface-hover transition-colors"
        title="Reset to start"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.992 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
        </svg>
      </button>
    </div>
  );
}
