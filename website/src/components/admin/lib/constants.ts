/** High-value memory types used for signal ratio calculations.
 *  Shared between ReportCard grading and API-level memory insights. */
export const SIGNAL_TYPES = new Set([
  "decision", "lesson_learned", "error_pattern", "user_preference",
  "user_fact", "reminder", "task_completion", "constraint",
  "advisor_insight",
]);
