import React, { useState, useEffect } from 'react';
import islGuideData from '../data/isl_sign_guide.json';
import { GuideCard, ISLSignEntry } from '../components/GuideCard';

interface LearnPageProps {
  onPracticeSign: (label: string) => void;
}

export const LearnPage: React.FC<LearnPageProps> = ({ onPracticeSign }) => {
  const [signs, setSigns] = useState<ISLSignEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('All');
  const [typeFilter, setTypeFilter] = useState('All');
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  useEffect(() => {
    setSigns(islGuideData.signs);
    const saved = localStorage.getItem('isl_favorites');
    if (saved) {
      try { setFavorites(new Set(JSON.parse(saved))); } catch(e) {}
    }
  }, []);

  const toggleFavorite = (label: string) => {
    const next = new Set(favorites);
    if (next.has(label)) next.delete(label);
    else next.add(label);
    setFavorites(next);
    localStorage.setItem('isl_favorites', JSON.stringify(Array.from(next)));
  };

  const filteredSigns = signs.filter(sign => {
    const matchesSearch = sign.displayName.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = categoryFilter === 'All' || categoryFilter === 'Favorites' ? true : sign.category === categoryFilter;
    const matchesType = typeFilter === 'All' ? true : sign.signType === typeFilter;
    const matchesFav = categoryFilter === 'Favorites' ? favorites.has(sign.label) : true;
    return matchesSearch && matchesCategory && matchesType && matchesFav;
  });

  const categories = ['All', 'Alphabet', 'Numbers', 'Words', 'Favorites'];
  const types = ['All', 'Static', 'Dynamic'];

  return (
    <div style={{ width: '100%', maxWidth: '1080px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ textAlign: 'center', marginBottom: '10px' }}>
        <h1 style={{ fontSize: '2rem', margin: '0 0 8px 0', color: 'var(--text-primary)' }}>Indian Sign Language Guide</h1>
        <p style={{ margin: 0, color: 'var(--text-secondary)' }}>Learn and reference genuine ISL hand signs before practicing.</p>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', background: 'var(--bg-card)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
        <div style={{ flex: 1, minWidth: '200px' }}>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>Search signs...</label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="e.g. water, hello, A..."
            style={{ width: '100%', padding: '8px 12px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)' }}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>Category</label>
          <div style={{ display: 'flex', gap: '8px' }}>
            {categories.map(cat => (
              <button key={cat} onClick={() => setCategoryFilter(cat)} style={{
                padding: '6px 12px', borderRadius: '4px', border: '1px solid var(--border-color)', cursor: 'pointer',
                background: categoryFilter === cat ? 'var(--accent-blue)' : 'var(--bg-card)',
                color: categoryFilter === cat ? '#fff' : 'var(--text-secondary)'
              }}>
                {cat}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>Sign Type</label>
          <div style={{ display: 'flex', gap: '8px' }}>
            {types.map(t => (
              <button key={t} onClick={() => setTypeFilter(t)} style={{
                padding: '6px 12px', borderRadius: '4px', border: '1px solid var(--border-color)', cursor: 'pointer',
                background: typeFilter === t ? 'var(--accent-blue)' : 'var(--bg-card)',
                color: typeFilter === t ? '#fff' : 'var(--text-secondary)'
              }}>
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
        {filteredSigns.map(sign => (
          <GuideCard
            key={sign.label}
            sign={sign}
            isFavorite={favorites.has(sign.label)}
            onToggleFavorite={toggleFavorite}
            onPracticeSign={onPracticeSign}
          />
        ))}
      </div>

      {filteredSigns.length === 0 && (
        <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-secondary)' }}>
          No signs found matching your search.
        </div>
      )}
    </div>
  );
};
