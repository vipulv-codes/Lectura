import React, { useEffect, useRef, useState } from 'react';
import { Play, Plus, BookOpen, Clock, AlertCircle, Trash2, Folder, FolderOpen, ChevronDown, ChevronRight, Loader2, RefreshCw } from 'lucide-react';
import { apiFetch, errorMessage } from '../lib/api';
import { formatDuration, isBusyStatus } from '../lib/format';
import type { VideoStatus } from '../lib/format';

export interface VideoMetadata {
  video_id: string;
  title: string;
  duration: number;
  author: string;
  thumbnail: string;
  url: string;
  status?: VideoStatus;
  error?: string;
  transcript_source?: string;
  n_chunks?: number;
}

interface IngestResponse {
  kind: 'video' | 'playlist';
  message: string;
  playlist?: { playlist_id: string };
}

export interface Playlist {
  playlist_id: string;
  title: string;
  videos: VideoMetadata[];
}

interface SidebarProps {
  videos: VideoMetadata[]; // Standalone videos
  playlists: Playlist[];
  whisperModel: string;
  onWhisperModelChange: (model: string) => void;
  activeVideoId: string | null;
  activePlaylistId: string | null;
  onSelectVideo: (videoId: string) => void;
  onSelectPlaylist: (playlistId: string) => void;
  onIngestSuccess: () => void;
  onDeleteVideo: (videoId: string) => void;
  onDeletePlaylist: (playlistId: string) => void;
  onRetryVideo: (videoId: string) => void;
}

// Label heuristic only; the backend decides what the URL actually is.
const looksLikePlaylist = (url: string) => /playlist\?/.test(url) || (/[?&]list=/.test(url) && !/[?&]v=/.test(url));

export const Sidebar: React.FC<SidebarProps> = ({
  videos,
  playlists,
  whisperModel,
  onWhisperModelChange,
  activeVideoId,
  activePlaylistId,
  onSelectVideo,
  onSelectPlaylist,
  onIngestSuccess,
  onDeleteVideo,
  onDeletePlaylist,
  onRetryVideo
}) => {
  const [url, setUrl] = useState('');
  const [isIngesting, setIsIngesting] = useState(false);
  const [statusStep, setStatusStep] = useState<'idle' | 'metadata' | 'success' | 'error'>('idle');
  const [errorText, setErrorText] = useState('');
  const [expandedPlaylists, setExpandedPlaylists] = useState<Record<string, boolean>>({});
  // Handle of the timer that hides the status box; cleared on unmount and on
  // every new submit so an old timer cannot reset a newer request's status.
  const resetTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(resetTimer.current), []);

  const scheduleReset = (ms: number) => {
    window.clearTimeout(resetTimer.current);
    resetTimer.current = window.setTimeout(() => setStatusStep('idle'), ms);
  };

  const togglePlaylist = (playlistId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedPlaylists(prev => ({
      ...prev,
      [playlistId]: !prev[playlistId]
    }));
  };

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || isIngesting) return;

    window.clearTimeout(resetTimer.current);
    setIsIngesting(true);
    setErrorText('');
    setStatusStep('metadata');

    try {
      // One endpoint; the backend distinguishes a video URL (even one carrying
      // a list= parameter) from a playlist URL.
      const data = await apiFetch<IngestResponse>('/api/ingest', {
        method: 'POST',
        body: { url: trimmed, whisper_model: whisperModel },
      });
      setStatusStep('success');
      setUrl('');
      onIngestSuccess();
      if (data.kind === 'playlist' && data.playlist?.playlist_id) {
        const id = data.playlist.playlist_id;
        setExpandedPlaylists(prev => ({ ...prev, [id]: true }));
      }
      scheduleReset(4000);
    } catch (err) {
      console.error(err);
      setStatusStep('error');
      setErrorText(errorMessage(err, 'Error occurred during ingestion'));
      scheduleReset(8000);
    } finally {
      setIsIngesting(false);
    }
  };

  const isPlaylistInput = looksLikePlaylist(url);

  return (
    <div className="sidebar">
      {/* Link Ingest Section */}
      <div className="sidebar-section">
        <h3>
          <Play size={16} />
          Ingest Link
        </h3>
        <form onSubmit={handleIngest} className="ingest-form">
          <input
            type="text"
            className="ingest-input"
            placeholder="Paste YouTube Video or Playlist Link..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={isIngesting}
          />
          <button 
            type="submit" 
            className="ingest-btn" 
            disabled={isIngesting || !url.trim()}
          >
            <Plus size={16} />
            {isIngesting ? 'Processing...' : isPlaylistInput ? 'Index Playlist' : 'Index Video'}
          </button>
          <label className="model-select-row" title="Only used when YouTube has no captions and audio must be transcribed locally. Smaller = faster, larger = more accurate.">
            <span>Transcription model</span>
            <select
              className="model-select"
              value={whisperModel}
              onChange={(e) => onWhisperModelChange(e.target.value)}
              disabled={isIngesting}
            >
              <option value="tiny.en">tiny.en — fastest (English)</option>
              <option value="tiny">tiny — fast (multilingual)</option>
              <option value="base">base — balanced</option>
              <option value="small">small — accurate (slow)</option>
            </select>
          </label>
        </form>

        {/* Status feedback for the (quick) registration step; the actual
            transcription/indexing runs in the background and its progress
            shows on each video's card in the library below. */}
        {statusStep !== 'idle' && (
          <div className="ingest-status-box">
            <div className={`status-step ${statusStep === 'success' ? 'completed' : statusStep === 'error' ? '' : 'active'}`}>
              {statusStep === 'success' ? '✓' : statusStep === 'error' ? '⚠️' : <div className="spinner-ring" />}
              {isPlaylistInput ? ' Registering playlist' : ' Adding to library'}
            </div>

            {statusStep === 'success' && (
              <div style={{ color: 'var(--success)', fontWeight: 'bold', fontSize: '11px', marginTop: '4px' }}>
                {isPlaylistInput
                  ? '🎉 Playlist registered! Videos are indexing in the background.'
                  : '🎉 Added! Indexing in the background — if it has no captions, local transcription can take a few minutes. Watch the card status.'
                }
              </div>
            )}
            {statusStep === 'error' && (
              <div style={{ color: 'var(--error)', display: 'flex', gap: '4px', marginTop: '4px' }}>
                <AlertCircle size={14} style={{ flexShrink: 0 }} />
                <span>{errorText}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Playlist & Lectures Library Section */}
      <div className="sidebar-section" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>
        <h3>
          <BookOpen size={16} />
          Lecture Library
        </h3>
        <div className="video-list-container">
          
          {/* Render Playlist Folders */}
          {playlists.map((pl) => {
            const isExpanded = !!expandedPlaylists[pl.playlist_id];
            const isFolderActive = activePlaylistId === pl.playlist_id && !activeVideoId;
            
            return (
              <div key={pl.playlist_id} className="playlist-folder-container">
                <div 
                  className={`playlist-folder-header ${isFolderActive ? 'active' : ''}`}
                  onClick={() => onSelectPlaylist(pl.playlist_id)}
                >
                  <button 
                    className="folder-toggle-btn"
                    onClick={(e) => togglePlaylist(pl.playlist_id, e)}
                  >
                    {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  </button>
                  <div className="folder-icon-wrapper">
                    {isExpanded ? <FolderOpen size={16} className="folder-icon" /> : <Folder size={16} className="folder-icon" />}
                  </div>
                  <div className="playlist-title" title={pl.title}>
                    {pl.title}
                  </div>
                  <button
                    className="playlist-delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`Are you sure you want to delete playlist "${pl.title}"? All associated videos and their indexes will also be deleted.`)) {
                        onDeletePlaylist(pl.playlist_id);
                      }
                    }}
                    title="Delete playlist folder"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>

                {isExpanded && (
                  <div className="playlist-nested-videos">
                    {pl.videos.length === 0 ? (
                      <div className="empty-folder-text">No videos in playlist.</div>
                    ) : (
                      pl.videos.map((vid) => {
                        const isVidActive = activeVideoId === vid.video_id;
                        const isTranscribing = vid.status === 'transcribing';
                        const isBusy = isBusyStatus(vid.status);
                        const isFailed = vid.status === 'failed';

                        return (
                          <div
                            key={vid.video_id}
                            className={`video-card nested ${isVidActive ? 'active' : ''} ${isBusy ? 'processing' : ''} ${isFailed ? 'failed' : ''}`}
                            onClick={() => !isBusy && !isFailed && onSelectVideo(vid.video_id)}
                          >
                            <div className="video-card-thumb-wrapper">
                              <img src={vid.thumbnail} alt={vid.title} className="video-card-thumb" />
                              {isBusy && (
                                <div className="thumb-processing-overlay">
                                  <Loader2 size={12} className="spinner pulse" />
                                </div>
                              )}
                            </div>
                            <div className="video-card-info">
                              <div className="video-card-title" title={vid.title}>
                                {vid.title}
                              </div>
                              <div className="video-card-meta">
                                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '100px' }}>
                                  {vid.author}
                                </span>
                                {isBusy ? (
                                  <span
                                    className="processing-badge"
                                    title={isTranscribing ? 'No captions found — transcribing audio locally with Whisper. This can take a few minutes.' : undefined}
                                  >
                                    {isTranscribing ? 'Transcribing…' : 'Processing…'}
                                  </span>
                                ) : isFailed ? (
                                  <span className="failed-badge" title={vid.error || 'Ingestion failed'}>
                                    <AlertCircle size={10} /> Failed
                                  </span>
                                ) : (
                                  <span style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                                    <Clock size={10} />
                                    {formatDuration(vid.duration)}
                                  </span>
                                )}
                              </div>
                            </div>
                            {isFailed && (
                              <button
                                className="video-retry-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onRetryVideo(vid.video_id);
                                }}
                                title={vid.error ? `Retry (last error: ${vid.error})` : 'Retry ingestion'}
                              >
                                <RefreshCw size={13} />
                              </button>
                            )}
                            <button
                              className="video-delete-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                if (confirm(`Are you sure you want to delete "${vid.title}"?`)) {
                                  onDeleteVideo(vid.video_id);
                                }
                              }}
                              title="Delete lecture"
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
            );
          })}

          {/* Standalone Videos */}
          {videos.length > 0 && (
            <div className="standalone-separator">
              <span>Standalone Lectures</span>
            </div>
          )}

          {videos.map((vid) => {
            const isVidActive = activeVideoId === vid.video_id && !activePlaylistId;
            const isTranscribing = vid.status === 'transcribing';
            const isBusy = isBusyStatus(vid.status);
            const isFailed = vid.status === 'failed';
            return (
              <div
                key={vid.video_id}
                className={`video-card ${isVidActive ? 'active' : ''} ${isBusy ? 'processing' : ''} ${isFailed ? 'failed' : ''}`}
                onClick={() => !isBusy && !isFailed && onSelectVideo(vid.video_id)}
              >
                <img src={vid.thumbnail} alt={vid.title} className="video-card-thumb" />
                <div className="video-card-info">
                  <div className="video-card-title" title={vid.title}>
                    {vid.title}
                  </div>
                  <div className="video-card-meta">
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '120px' }}>
                      {vid.author}
                    </span>
                    {isBusy ? (
                      <span
                        className="processing-badge"
                        title={isTranscribing ? 'No captions found — transcribing audio locally with Whisper. This can take a few minutes.' : undefined}
                      >
                        {isTranscribing ? 'Transcribing…' : 'Processing…'}
                      </span>
                    ) : isFailed ? (
                      <span className="failed-badge" title={vid.error || 'Ingestion failed'}>
                        <AlertCircle size={10} /> Failed
                      </span>
                    ) : (
                      <span style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                        <Clock size={10} />
                        {formatDuration(vid.duration)}
                      </span>
                    )}
                  </div>
                </div>
                {isFailed && (
                  <button
                    className="video-retry-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRetryVideo(vid.video_id);
                    }}
                    title={vid.error ? `Retry (last error: ${vid.error})` : 'Retry ingestion'}
                  >
                    <RefreshCw size={13} />
                  </button>
                )}
                <button
                  className="video-delete-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm(`Are you sure you want to delete "${vid.title}"? This cannot be undone.`)) {
                      onDeleteVideo(vid.video_id);
                    }
                  }}
                  title="Delete lecture"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            );
          })}

          {playlists.length === 0 && videos.length === 0 && (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px', textAlign: 'center', marginTop: '20px' }}>
              No lectures indexed yet.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
