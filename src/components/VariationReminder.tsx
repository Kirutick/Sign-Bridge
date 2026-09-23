import React, { useState, useEffect } from "react";

const REMINDERS = [
  "💡 Try shifting your hand slightly left or right",
  "💡 Vary your signing speed (slightly faster or slower)",
  "💡 Move hand slightly closer or farther from the camera",
  "💡 Tilt your wrist or palm orientation slightly",
  "💡 Ensure fingers execute full natural gesture articulation",
];

export const VariationReminder: React.FC = () => {
  const [index, setIndex] = useState<number>(0);

  useEffect(() => {
    // Non-repeating rotation on interval (§7)
    setIndex((prev) => (prev + 1) % REMINDERS.length);
  }, []);

  return (
    <div
      style={{
        padding: "8px 14px",
        background: "rgba(0, 0, 0, 0.2)",
        border: "1px dashed var(--border-color)",
        borderRadius: "var(--radius-sm)",
        color: "var(--text-secondary)",
        fontSize: "0.8rem",
        fontStyle: "italic",
      }}
    >
      {REMINDERS[index]}
    </div>
  );
};
