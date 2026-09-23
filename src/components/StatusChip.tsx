import React from "react";
import styles from "../styles/StatusChip.module.css";

interface StatusChipProps {
  icon: string;
  label: string;
  value: string;
  variant?: "active" | "warning" | "error" | "idle";
}

export const StatusChip: React.FC<StatusChipProps> = ({
  icon,
  label,
  value,
  variant = "idle",
}) => {
  return (
    <div className={`${styles.chip} ${styles[variant]}`} role="status">
      <span className={styles.dot} />
      <span>{icon}</span>
      <span>
        {label}: <strong>{value}</strong>
      </span>
    </div>
  );
};
