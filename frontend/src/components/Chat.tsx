import React, { useState, useEffect, useRef } from 'react';
import { Send, GraduationCap, User, Loader, Trash2 } from 'lucide-react';
import { apiFetch, errorMessage } from '../lib/api';

// A retrieved transcript chunk returned alongside an answer (see backend
// lectura.generation.Source).
export interface Source {
  timestamp: string;
  seconds: number;
  video_id: string;
  video_title: string;
  snippet?: string;
  score?: number;
  chunk_id?: string;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
  abstained?: boolean;
}

export interface ChatMessage {
  role: 'student' | 'assistant';
  text: string;
  sources?: Source[];
}

interface ChatProps {
  videoId?: string | null;
  playlistId?: string | null;
  apiKey: string;
  onTimestampClick: (seconds: number, videoId?: string) => void;
  messages: ChatMessage[];
  onMessagesChange: (newMessages: ChatMessage[]) => void;
  onClearChat: () => void;
}

export const Chat: React.FC<ChatProps> = ({
  videoId,
  playlistId,
  apiKey,
  onTimestampClick,
  messages,
  onMessagesChange,
  onClearChat
}) => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Resolve which retrieved source (video) a cited timestamp belongs to.
  const resolveSource = (prefix: string | undefined, seconds: number, msgSources?: Source[]): Source | null => {
    if (!msgSources || msgSources.length === 0) return null;
    let matchSrc: Source | undefined;
    if (prefix) {
      const cleanPrefix = prefix.trim().toLowerCase();
      // 1. Match prefix (lecture title) and timestamp range together
      matchSrc = msgSources.find(s => {
        const title = s.video_title?.toLowerCase() || '';
        return title.includes(cleanPrefix) && (Math.abs(s.seconds - seconds) < 5);
      });
      // 2. Match prefix only
      if (!matchSrc) {
        matchSrc = msgSources.find(s => s.video_title?.toLowerCase().includes(cleanPrefix));
      }
    }
    // 3. Time-based match
    if (!matchSrc) {
      matchSrc = msgSources.find(s => Math.abs(s.seconds - seconds) < 3);
    }
    // 4. Nearest-source fallback: cited line timestamps can sit anywhere
    // inside a retrieved segment, so pick the source closest in time. This
    // keeps playlist citations resolving to a video instead of dead-ending.
    if (!matchSrc) {
      matchSrc = msgSources.reduce((best, s) =>
        Math.abs(s.seconds - seconds) < Math.abs(best.seconds - seconds) ? s : best
      );
    }
    return matchSrc ?? null;
  };

  // Citations closer together than this are collapsed into the first one.
  // Caption lines are only a few seconds long, so models tend to cite several
  // consecutive lines of one continuous explanation; one link is enough.
  const CITATION_COLLAPSE_SECS = 15;

  // Helper to parse timestamps in text and turn them into clickable buttons.
  // Handles single citations like [12:34] / (1:02:03), title-prefixed ones
  // like [Lecture 13, 23:30], and comma-separated lists like
  // [02:23:22, 02:23:46, 02:24:49] — each timestamp becomes its own button,
  // with near-duplicate chains collapsed to the first timestamp.
  const parseTimestamps = (text: string, msgSources?: Source[]): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    // Seconds of the most recently emitted citation, used for collapsing
    // chains like "[a], [b] and [c]" that are separated only by connectors.
    let lastEmitted = Number.NEGATIVE_INFINITY;

    // Any bracket/paren group containing at least one MM:SS / HH:MM:SS stamp.
    const groupRegex = /[[(]([^\])]*?\d{1,2}:\d{2}[^\])]*)[\])]/g;
    const tsRegex = /(\d{1,2}):(\d{2})(?::(\d{2}))?/g;
    let match: RegExpExecArray | null;

    while ((match = groupRegex.exec(text)) !== null) {
      const inner = match[1];
      const stamps = Array.from(inner.matchAll(tsRegex));
      if (stamps.length === 0) continue;

      const betweenText = match.index > lastIndex ? text.substring(lastIndex, match.index) : '';
      // Only pure connectors (commas, "and", whitespace) chain a collapse
      // across groups; real prose in between starts a fresh citation context.
      const isConnector = /^[\s,;]*(?:and[\s,;]*)?$/i.test(betweenText);

      const open = match[0][0];
      const close = match[0][match[0].length - 1];
      // Text before the first timestamp is a title prefix, e.g. "Lecture 13".
      const rawPrefix = inner.slice(0, stamps[0].index ?? 0);
      const prefix = rawPrefix.replace(/(?:,|:|@|\bat\b|\s)+$/i, '').trim() || undefined;

      // Convert stamps and drop the ones that sit within the collapse window
      // of the previous (kept or skipped) one — chains fold into their start.
      const survivors: { label: string; seconds: number }[] = [];
      let prev = isConnector ? lastEmitted : Number.NEGATIVE_INFINITY;
      for (const m of stamps) {
        const a = parseInt(m[1], 10);
        const b = parseInt(m[2], 10);
        const seconds = m[3] !== undefined
          ? a * 3600 + b * 60 + parseInt(m[3], 10)  // HH:MM:SS
          : a * 60 + b;                             // MM:SS
        if (Math.abs(seconds - prev) <= CITATION_COLLAPSE_SECS) {
          prev = seconds;
          continue;
        }
        survivors.push({ label: `${open}${m[0]}${close}`, seconds });
        prev = seconds;
      }

      if (survivors.length === 0) {
        // Whole group was redundant (can only happen after a connector):
        // swallow the connector too so no dangling ", " is left behind.
        lastIndex = groupRegex.lastIndex;
        continue;
      }

      if (betweenText) parts.push(betweenText);
      if (prefix) parts.push(`${prefix} `);
      survivors.forEach((s, i) => {
        const src = resolveSource(prefix, s.seconds, msgSources);
        if (i > 0) parts.push(' ');
        parts.push(
          <button
            key={`${match!.index}-${i}`}
            className="timestamp-link"
            onClick={() => onTimestampClick(s.seconds, src?.video_id)}
            title={src?.video_title ? `Jump to "${src.video_title}" at ${s.label}` : `Jump to ${s.label}`}
          >
            {s.label}
          </button>
        );
      });
      lastEmitted = survivors[survivors.length - 1].seconds;

      lastIndex = groupRegex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : [text];
  };

  // Helper to process simple markdown bolding and lists
  const formatMessageText = (text: string, msgSources?: Source[]): React.ReactNode => {
    return (text ?? '').split('\n').map((line, idx) => {
      let content: React.ReactNode[];

      // Simple bold text parsing (**text**)
      if (line.includes('**')) {
        const boldRegex = /\*\*(.*?)\*\*/g;
        const boldParts: React.ReactNode[] = [];
        let lastBIndex = 0;
        let bMatch: RegExpExecArray | null;

        while ((bMatch = boldRegex.exec(line)) !== null) {
          if (bMatch.index > lastBIndex) {
            boldParts.push(...parseTimestamps(line.substring(lastBIndex, bMatch.index), msgSources));
          }
          // Parse timestamps inside the bold span too, so citations like
          // **[12:34]** stay clickable.
          boldParts.push(<strong key={bMatch.index} style={{ color: 'var(--text-primary)' }}>{parseTimestamps(bMatch[1], msgSources)}</strong>);
          lastBIndex = boldRegex.lastIndex;
        }
        
        if (lastBIndex < line.length) {
          boldParts.push(...parseTimestamps(line.substring(lastBIndex), msgSources));
        }
        content = boldParts;
      } else {
        content = parseTimestamps(line, msgSources);
      }

      // Check if list item
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        // Skip bullet mark in output
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
        return (
          <li key={idx} style={{ marginLeft: '20px', listStyleType: 'disc', marginBottom: '4px' }}>
            {displayContent}
          </li>
        );
      }
      
      // Check if header
      if (line.trim().startsWith('###')) {
        return <h4 key={idx} style={{ marginTop: '10px', marginBottom: '4px', fontWeight: 'bold', color: '#c084fc' }}>{content}</h4>;
      }
      if (line.trim().startsWith('##')) {
        return <h3 key={idx} style={{ marginTop: '14px', marginBottom: '6px', fontWeight: 'bold', fontSize: '15px', color: '#a78bfa' }}>{content}</h3>;
      }
      
      return <p key={idx} style={{ minHeight: '1.2em', marginBottom: '6px' }}>{content}</p>;
    });
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (!apiKey.trim()) {
      alert('Please enter your Gemini API Key in the top header first.');
      return;
    }

    const studentMessage = input.trim();
    const updatedWithStudent: ChatMessage[] = [...messages, { role: 'student', text: studentMessage }];
    onMessagesChange(updatedWithStudent);
    setInput('');
    setIsLoading(true);

    try {
      const data = await apiFetch<QueryResponse>('/api/query', {
        method: 'POST',
        headers: { 'X-Gemini-API-Key': apiKey.trim() },
        body: {
          video_id: videoId || undefined,
          playlist_id: playlistId || undefined,
          query: studentMessage,
        },
      });
      onMessagesChange([...updatedWithStudent, { role: 'assistant', text: data.answer, sources: data.sources }]);
    } catch (err) {
      console.error(err);
      const message = errorMessage(err, 'Make sure the backend is running and your API key is valid.');
      onMessagesChange([
        ...updatedWithStudent,
        {
          role: 'assistant',
          text: `⚠️ Error processing query: ${message}`
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Sub-header for chat session details and actions */}
      <div className="chat-header">
        <span>Session History</span>
        <button type="button" className="chat-clear-btn" onClick={onClearChat} title="Clear Chat History">
          <Trash2 size={13} />
          <span>Clear Chat</span>
        </button>
      </div>

      {/* Chat History Panel */}
      <div className="chat-history">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'student' ? <User size={16} /> : <GraduationCap size={16} />}
            </div>
            <div className="message-bubble">
              {formatMessageText(msg.text, msg.sources)}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="chat-message assistant">
            <div className="message-avatar">
              <Loader size={16} className="spinner" />
            </div>
            <div className="message-bubble" style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>Thinking... Searching library segments...</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Chat Input Bar */}
      <form onSubmit={handleSend} className="chat-input-bar">
        <input
          type="text"
          className="chat-input"
          placeholder={apiKey ? "Ask a question about the active lectures..." : "⚠️ Please enter API key above first"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading || !apiKey}
        />
        <button
          type="submit"
          className="chat-send-btn"
          disabled={isLoading || !input.trim() || !apiKey}
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
};
