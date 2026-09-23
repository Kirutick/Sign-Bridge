import React, { useState, useEffect, useMemo } from 'react';
import styles from '../styles/QuickSignGuide.module.css';
import islGuideData from '../data/isl_sign_guide.json';

interface QuickSignGuideProps {
  onPracticeSign: (label: string) => void;
  onOpenFullGuide: () => void;
}

export const QuickSignGuide: React.FC<QuickSignGuideProps> = ({ onPracticeSign, onOpenFullGuide }) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(() => {
    const saved = localStorage.getItem('quickGuideExpanded');
    return saved !== null ? JSON.parse(saved) : true;
  });
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    localStorage.setItem('quickGuideExpanded', JSON.stringify(isExpanded));
  }, [isExpanded]);

  const toggleExpand = () => setIsExpanded(!isExpanded);

  const { exactMatch, spellingMode, spelledLetters } = useMemo(() => {
    if (!searchQuery.trim()) return { exactMatch: null, spellingMode: false, spelledLetters: [] };

    const query = searchQuery.trim().toLowerCase();
    
    // Exact match search
    const exact = islGuideData.signs.find(s => 
      s.label.toLowerCase() === query.replace(/\s+/g, '_') ||
      s.displayName.toLowerCase() === query
    );

    if (exact) {
      return { exactMatch: exact, spellingMode: false, spelledLetters: [] };
    }

    // Spelling mode fallback for non-empty queries that don't exact match
    const letters = query.replace(/[^a-z]/g, '').toUpperCase().split('');
    const spelled = letters.map(char => islGuideData.signs.find(s => s.label === char));

    return { exactMatch: null, spellingMode: true, spelledLetters: spelled };
  }, [searchQuery]);

  return (
    <div className={`${styles.container} ${!isExpanded ? styles.containerCollapsed : ''}`}>
      <div className={styles.header} onClick={toggleExpand}>
        <div className={styles.headerTitle}>
          <span className={styles.headerIcon}>🔎</span>
          {isExpanded ? 'Search ISL signs...' : 'ISL Sign Guide'}
        </div>
        <button className={styles.toggleBtn}>
          {isExpanded ? '▼' : '▲'}
        </button>
      </div>

      {isExpanded && (
        <div className={styles.content}>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Type a word or letter..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            autoFocus
          />

          {exactMatch && (
            <div className={styles.resultArea}>
              <div className={styles.resultHeader}>
                <span className={styles.resultTitle}>{exactMatch.displayName}</span>
              </div>
              
              <div className={styles.resultImageContainer}>
                {exactMatch.reference ? (
                  <img src={exactMatch.reference} alt={`ISL sign for ${exactMatch.displayName}`} className={styles.resultImage} />
                ) : (
                  <span className={styles.noImage}>📹</span>
                )}
              </div>

              <div className={styles.actions}>
                <button className={`${styles.actionBtn} ${styles.secondaryBtn}`} onClick={onOpenFullGuide}>
                  Open Full Guide
                </button>
                <button className={`${styles.actionBtn} ${styles.primaryBtn}`} onClick={() => onPracticeSign(exactMatch.label)}>
                  Practice
                </button>
              </div>
            </div>
          )}

          {spellingMode && spelledLetters.length > 0 && (
            <div className={styles.spellingContainer}>
              <div className={styles.spellingMessage}>
                No direct ISL reference available.<br/>
                You can spell this word using the ISL alphabet:
              </div>
              <div className={styles.alphabetGrid}>
                {spelledLetters.map((sign, idx) => sign && (
                  <div key={idx} className={styles.alphabetItem}>
                    {sign.reference ? (
                       <img src={sign.reference} alt={sign.label} className={styles.alphabetImage} />
                    ) : (
                       <div className={styles.alphabetImage} style={{display:'flex', alignItems:'center', justifyContent:'center'}}>📹</div>
                    )}
                    <span className={styles.alphabetLetter}>{sign.label}</span>
                  </div>
                ))}
              </div>
              <div className={styles.actions} style={{ marginTop: '8px' }}>
                 <button className={`${styles.actionBtn} ${styles.secondaryBtn}`} onClick={onOpenFullGuide}>
                   Open Full Guide
                 </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
