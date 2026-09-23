import React from 'react';

export interface ISLSignEntry {
  label: string;
  displayName: string;
  modelClassIndex: number;
  category: string;
  signType: string;
  verificationStatus: string;
  reference: string | null;
  instructions: string[];
}

interface GuideCardProps {
  sign: ISLSignEntry;
  isFavorite: boolean;
  onToggleFavorite: (label: string) => void;
  onPracticeSign: (label: string) => void;
}

export const GuideCard: React.FC<GuideCardProps> = ({ sign, isFavorite, onToggleFavorite, onPracticeSign }) => {
  const isVerified = sign.verificationStatus === 'verified';

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-md)',
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>
            {sign.displayName}
          </h2>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px', fontWeight: 600 }}>
            Indian Sign Language • {sign.signType} Sign
          </div>
        </div>
        <button
          onClick={() => onToggleFavorite(sign.label)}
          style={{
            background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.5rem',
            opacity: isFavorite ? 1 : 0.3,
            transition: 'opacity 0.2s'
          }}
          title={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
        >
          ⭐
        </button>
      </div>

      <div style={{
        width: '100%', height: '180px',
        background: 'rgba(0,0,0,0.3)',
        borderRadius: 'var(--radius-sm)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid var(--border-color)',
        overflow: 'hidden',
        position: 'relative'
      }}>
        {sign.reference ? (
           <img src={sign.reference} alt={`ISL reference for ${sign.displayName}`} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
        ) : (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <div style={{ fontSize: '2rem', opacity: 0.5 }}>📹</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '8px' }}>Reference unavailable</div>
          </div>
        )}
      </div>

      {!isVerified && (
        <div style={{
          padding: '8px 12px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.4)',
          borderRadius: '4px', color: '#f59e0b', fontSize: '0.8rem', fontWeight: 600
        }}>
          ⚠️ This sign has not yet been verified for the Sign Bridge guide.
        </div>
      )}

      {sign.instructions.length > 0 && (
        <div style={{ flex: 1 }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--accent-cyan)', marginBottom: '8px' }}>How to perform:</h3>
          <ol style={{ margin: 0, paddingLeft: '20px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            {sign.instructions.map((step, idx) => (
              <li key={idx} style={{ marginBottom: '4px' }}>{step}</li>
            ))}
          </ol>
        </div>
      )}

      <div style={{ marginTop: 'auto', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {sign.reference ? 'Source: Verified Reference' : ''}
        </div>
        <button
          onClick={() => onPracticeSign(sign.label)}
          style={{
            background: 'var(--accent-blue)', color: '#fff', border: 'none', padding: '8px 16px',
            borderRadius: 'var(--radius-sm)', fontWeight: 700, cursor: 'pointer', fontSize: '0.85rem'
          }}
        >
          Practice This Sign
        </button>
      </div>
    </div>
  );
};
