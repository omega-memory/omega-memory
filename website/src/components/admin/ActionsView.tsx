import { useEffect, useState } from "react";
import ActionsHeader from "./ActionsHeader";
import TweetReview from "./TweetReview";
import AutonomousNotice from "./AutonomousNotice";
import type { XAccount } from "./lib/types";

interface ActionsViewProps {
  onCountChange: (count: number) => void;
  onGenerate?: () => void;
  generating?: boolean;
  genProgress?: string;
  refreshKey?: number;
  account?: XAccount;
  onAccountChange?: (account: XAccount) => void;
}

export default function ActionsView({
  onCountChange,
  onGenerate,
  generating,
  genProgress,
  refreshKey,
  account = "jasonsosa",
  onAccountChange,
}: ActionsViewProps) {
  const [reviewCount, setReviewCount] = useState(0);

  // Force zero count when omega_memory is selected (autonomous, no queue)
  useEffect(() => {
    if (account === "omega_memory") {
      setReviewCount(0);
      onCountChange(0);
    }
  }, [account, onCountChange]);

  useEffect(() => {
    onCountChange(reviewCount);
  }, [reviewCount, onCountChange]);

  return (
    <div>
      {/* Header: generate button + account context */}
      <ActionsHeader
        account={account}
        onAccountChange={onAccountChange ?? (() => {})}
        onGenerate={onGenerate}
        generating={generating}
        genProgress={genProgress}
      />

      {/* Content: tweet review queue or autonomous status */}
      <div className="admin-tab-enter">
        {account === "omega_memory" ? (
          <AutonomousNotice />
        ) : (
          <TweetReview
            onCountChange={setReviewCount}
            refreshKey={refreshKey}
            account={account}
          />
        )}
      </div>
    </div>
  );
}
