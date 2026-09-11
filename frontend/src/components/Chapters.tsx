import React, { useState } from 'react';
import { BookOpen, FileText, Loader, Play, Trash2, Download } from 'lucide-react';
import { apiFetch, errorMessage } from '../lib/api';

interface ChaptersProps {
  videoId: string;
  videoTitle: string;
  apiKey: string;
  onTimestampClick: (seconds: number, videoId?: string) => void;
  chaptersMarkdown: string | null;
  onChaptersChange: (markdown: string | null) => void;
}

export const Chapters: React.FC<ChaptersProps> = ({
  videoId,
  videoTitle,
  apiKey,
  onTimestampClick,
  chaptersMarkdown,
  onChaptersChange
}) => {
  const [isGenerating, setIsGenerating] = useState(false);

  const handleDownloadMarkdown = () => {
    if (!chaptersMarkdown) return;
    const blob = new Blob([chaptersMarkdown], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    
    // Sanitize filename to remove characters that are invalid in Windows/macOS filesystems
    const safeTitle = videoTitle.replace(/[/\\?%*:|"<>]/g, '-').trim();
    link.setAttribute('download', `${safeTitle}_Summary.md`);
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleClearOutline = () => {
    const confirmClear = window.confirm("Are you sure you want to clear this chapter outline? This will free up storage space and allow you to regenerate it later.");
    if (confirmClear) {
      onChaptersChange(null);
    }
  };

  const generateOutline = async () => {
    if (!apiKey.trim()) {
      alert('Please enter your Gemini API Key in the top header first.');
      return;
    }

    setIsGenerating(true);
    try {
      const data = await apiFetch<{ chapters: string }>('/api/chapters', {
        method: 'POST',
        headers: { 'X-Gemini-API-Key': apiKey.trim() },
        body: { video_id: videoId },
      });
      if (!data.chapters?.trim()) throw new Error('The model returned an empty outline.');
      onChaptersChange(data.chapters);
    } catch (err) {
      console.error(err);
      alert(`Error generating outline: ${errorMessage(err, 'Unknown error')}`);
    } finally {
      setIsGenerating(false);
    }
  };

  // Render timestamps inside text as clickable buttons
  const parseTimestampsInMarkdown = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    
    // Regex for [MM:SS] or [HH:MM:SS]
    const regex = /\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]/g;
    let match: RegExpExecArray | null;
    
    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      
      const fullMatch = match[0];
      const val1 = parseInt(match[1], 10);
      const val2 = parseInt(match[2], 10);
      const val3 = match[3] ? parseInt(match[3], 10) : undefined;
      
      let seconds = 0;
      if (val3 !== undefined) {
        seconds = val1 * 3600 + val2 * 60 + val3;
      } else {
        seconds = val1 * 60 + val2;
      }
      
      parts.push(
        <button
          key={match.index}
          className="timestamp-link"
          onClick={() => onTimestampClick(seconds)}
          title={`Seek to ${fullMatch}`}
          style={{ padding: '0px 4px', fontSize: '11px', verticalAlign: 'middle' }}
        >
          <Play size={8} style={{ marginRight: '3px', display: 'inline-block' }} />
          {fullMatch}
        </button>
      );
      
      lastIndex = regex.lastIndex;
    }
    
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts.length > 0 ? parts : [text];
  };

  // Convert raw markdown outline from Gemini into HTML tags
  const renderMarkdown = (markdown: string) => {
    const lines = (markdown ?? '').split('\n');
    let insideTable = false;
    let tableHeaders: string[] = [];
    let tableRows: string[][] = [];

    const elements: React.ReactNode[] = [];

    lines.forEach((line, idx) => {
      const trimmed = line.trim();

      // Handle Tables
      if (trimmed.startsWith('|')) {
        // Skip separator lines e.g. |---|---|
        if (trimmed.includes('---')) {
          return;
        }

        const cells = trimmed
          .split('|')
          .map(c => c.trim())
          .filter((_, i, arr) => i > 0 && i < arr.length - 1); // remove outer empty items from split

        if (!insideTable) {
          insideTable = true;
          tableHeaders = cells;
          tableRows = [];
        } else {
          tableRows.push(cells);
        }
        return;
      }

      // If table ended, render it
      if (insideTable && !trimmed.startsWith('|')) {
        insideTable = false;
        elements.push(
          <table key={`table-${idx}`}>
            <thead>
              <tr>
                {tableHeaders.map((h, i) => <th key={i}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row, rIdx) => (
                <tr key={rIdx}>
                  {row.map((cell, cIdx) => (
                    <td key={cIdx}>{parseTimestampsInMarkdown(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        );
      }

      // Skip empty lines
      if (!trimmed) return;

      // Check Bold matches (**text**)
      let content: React.ReactNode[];
      if (line.includes('**')) {
        const boldRegex = /\*\*(.*?)\*\*/g;
        const boldParts: React.ReactNode[] = [];
        let lastBIndex = 0;
        let bMatch: RegExpExecArray | null;
        while ((bMatch = boldRegex.exec(line)) !== null) {
          if (bMatch.index > lastBIndex) {
            boldParts.push(...parseTimestampsInMarkdown(line.substring(lastBIndex, bMatch.index)));
          }
          // Keep timestamps inside bold spans clickable (e.g. **[12:34]**).
          boldParts.push(<strong key={bMatch.index} style={{ color: 'var(--text-primary)' }}>{parseTimestampsInMarkdown(bMatch[1])}</strong>);
          lastBIndex = boldRegex.lastIndex;
        }
        if (lastBIndex < line.length) {
          boldParts.push(...parseTimestampsInMarkdown(line.substring(lastBIndex)));
        }
        content = boldParts;
      } else {
        content = parseTimestampsInMarkdown(line);
      }

      // Headers
      if (trimmed.startsWith('# ')) {
        elements.push(<h1 key={idx} style={{ color: '#a78bfa', marginTop: '24px', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>{content}</h1>);
      } else if (trimmed.startsWith('## ')) {
        elements.push(<h2 key={idx} style={{ color: '#c084fc', marginTop: '20px' }}>{content}</h2>);
      } else if (trimmed.startsWith('### ')) {
        elements.push(<h3 key={idx} style={{ color: '#a78bfa', marginTop: '16px' }}>{content}</h3>);
      }
      // Bullet list items
      else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
        let displayContent = content;
        if (Array.isArray(content) && content.length > 0) {
          const first = content[0];
          if (typeof first === 'string') {
            const trimmed = first.trimStart();
            if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
              displayContent = [trimmed.substring(2), ...content.slice(1)];
            }
          }
        }
        elements.push(
          <li key={idx} style={{ marginLeft: '24px', listStyleType: 'disc', color: 'var(--text-secondary)', marginBottom: '6px' }}>
            {displayContent}
          </li>
        );
      }
      // Ordered lists
      else if (/^\d+\.\s/.test(trimmed)) {
        elements.push(
          <li key={idx} style={{ marginLeft: '24px', listStyleType: 'decimal', color: 'var(--text-secondary)', marginBottom: '6px' }}>
            {content}
          </li>
        );
      }
      // Paragraphs
      else {
        elements.push(<p key={idx} style={{ marginBottom: '12px', color: 'var(--text-secondary)' }}>{content}</p>);
      }
    });

    // Render final trailing table if open
    if (insideTable) {
      elements.push(
        <table key="table-final">
          <thead>
            <tr>
              {tableHeaders.map((h, i) => <th key={i}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {tableRows.map((row, rIdx) => (
              <tr key={rIdx}>
                {row.map((cell, cIdx) => (
                  <td key={cIdx}>{parseTimestampsInMarkdown(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
    }

    return elements;
  };

  return (
    <div className="chapters-container">
      {/* Sub-header for outline actions */}
      <div className="chapters-header">
        <span>Lecture Timeline Outline</span>
        {chaptersMarkdown && (
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              className="chapters-action-btn download"
              onClick={handleDownloadMarkdown}
              title="Download Outline as Markdown (.md)"
            >
              <Download size={13} />
              <span>Download Outline</span>
            </button>
            <button
              type="button"
              className="chapters-action-btn clear"
              onClick={handleClearOutline}
              title="Clear Lecture Outline"
            >
              <Trash2 size={13} />
              <span>Clear Outline</span>
            </button>
          </div>
        )}
      </div>

      {chaptersMarkdown ? (
        <div className="chapters-content">
          {renderMarkdown(chaptersMarkdown)}
        </div>
      ) : (
        <div className="generate-chapters-overlay">
          <FileText size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
          <h2 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '8px' }}>Generate Lecture Notes & Chapters</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '13.5px', maxWidth: '400px', margin: '0 auto 16px auto', lineHeight: '1.4' }}>
            Have Gemini scan the entire lecture transcript to build a structured timeline table of contents and summary notes.
          </p>
          <button 
            className="gen-chapters-btn"
            onClick={generateOutline}
            disabled={isGenerating}
          >
            {isGenerating ? (
              <>
                <Loader size={16} className="spinner" />
                Processing Transcript...
              </>
            ) : (
              <>
                <BookOpen size={16} />
                Generate Outline
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
