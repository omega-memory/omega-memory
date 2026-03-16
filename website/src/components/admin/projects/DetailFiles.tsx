import type { FileClaimItem } from "./types";
import { relativeTime } from "./utils";

interface DetailFilesProps {
  fileClaims: FileClaimItem[];
}

export default function DetailFiles({ fileClaims }: DetailFilesProps) {
  if (fileClaims.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
        Files in Play
      </h3>
      <div className="space-y-1 max-h-40 overflow-y-auto">
        {fileClaims.map((claim, i) => {
          const basename = claim.filePath.split("/").pop() ?? claim.filePath;
          return (
            <div
              key={`${claim.filePath}-${i}`}
              className="flex items-center gap-2 rounded px-2 py-1 text-xs group"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-amber-400/40 flex-shrink-0" />
              <span
                className="text-white/50 truncate font-mono"
                title={claim.filePath}
              >
                {basename}
              </span>
              <span className="ml-auto text-[10px] text-white/20 flex-shrink-0">
                {relativeTime(claim.claimedAt)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
