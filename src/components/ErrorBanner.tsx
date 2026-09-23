import React from "react";
import styles from "../styles/ErrorBanner.module.css";

interface ErrorBannerProps {
  title: string;
  message: string;
  onRetry?: () => void;
  retryText?: string;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  title,
  message,
  onRetry,
  retryText = "🔄 Retry Action",
}) => {
  return (
    <div className={styles.banner} role="alert">
      <div className={styles.content}>
        <span className={styles.icon}>⚠️</span>
        <div>
          <div className={styles.title}>{title}</div>
          <div>{message}</div>
        </div>
      </div>

      {onRetry && (
        <button type="button" className={styles.retryBtn} onClick={onRetry}>
          {retryText}
        </button>
      )}
    </div>
  );
};
