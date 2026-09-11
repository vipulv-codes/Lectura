import { useCallback, useEffect, useRef, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import type { VideoMetadata, Playlist } from './components/Sidebar';
import { KeyInput } from './components/KeyInput';
import { VideoPlayer } from './components/VideoPlayer';
import type { SeekRequest } from './components/VideoPlayer';
import { Chat } from './components/Chat';
import type { ChatMessage, Source } from './components/Chat';
import { Chapters } from './components/Chapters';
import { GraduationCap, Sparkles, MessageSquare, BookOpen, FolderOpen, Play, Loader2, ArrowLeft, X, RefreshCw, AlertCircle } from 'lucide-react';
import { apiFetch, errorMessage } from './lib/api';
import { storage } from './lib/storage';
import { formatDuration, isBusyStatus } from './lib/format';
import './App.css';

const KEYS = {
  chats: 'lectura_chat_histories',
  chapters: 'lectura_chapter_outlines',
  activeVideo: 'lectura_active_video_id',
  activePlaylist: 'lectura_active_playlist_id',
  whisper: 'lectura_whisper_model',
} as const;

const POLL_MS = 4000;

const chatKey = (id: string, isPlaylist: boolean) => (isPlaylist ? `playlist_${id}` : `video_${id}`);

const omitKeys = <T,>(obj: Record<string, T>, keys: string[]): Record<string, T> => {
  const drop = new Set(keys);
  return Object.fromEntries(Object.entries(obj).filter(([k]) => !drop.has(k)));
};

// Retrieved snippets are never rendered; keep them out of localStorage so the
// chat history does not grow by kilobytes per question.
const withoutSnippet = (source: Source): Source => {
  const copy = { ...source };
  delete copy.snippet;
  return copy;
};

const stripForStorage = (histories: Record<string, ChatMessage[]>): Record<string, ChatMessage[]> =>
  Object.fromEntries(
    Object.entries(histories).map(([k, msgs]) => [
      k,
      msgs.map((m) => (m.sources ? { ...m, sources: m.sources.map(withoutSnippet) } : m)),
    ]),
  );

const welcome = (isPlaylist: boolean): ChatMessage[] => [
  {
    role: 'assistant',
    text: isPlaylist
      ? "Hi! I am your AI Lecture Teaching Assistant. Ask me anything across all the videos in this playlist folder, and I'll search their transcripts to find the answers with clickable timestamps!"
      : "Hi! I am your AI Lecture Teaching Assistant. Ask me anything about this lecture, and I'll find the answers along with direct timestamped links!",
  },
];

function App() {
  const [apiKey, setApiKey] = useState<string>('');
  const [allVideos, setAllVideos] = useState<VideoMetadata[]>([]);
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [activeVideoId, setActiveVideoId] = useState<string | null>(() => storage.get(KEYS.activeVideo));
  const [activePlaylistId, setActivePlaylistId] = useState<string | null>(() => storage.get(KEYS.activePlaylist));
  const [seek, setSeek] = useState<SeekRequest>(null);
  const seekNonce = useRef(0);
  const [activeTab, setActiveTab] = useState<'chat' | 'chapters'>('chat');
  // Whisper model for the local-transcription fallback. Lives here (not in the
  // Sidebar) so the retry flow can reuse the user's current selection.
  const [whisperModel, setWhisperModel] = useState<string>(() => storage.get(KEYS.whisper) || 'tiny.en');
  // Videos currently being re-ingested via the retry button, shown as
  // 'processing' until the refreshed library reflects the real status.
  const [retryingIds, setRetryingIds] = useState<Set<string>>(new Set());
  const [chatHistories, setChatHistories] = useState<Record<string, ChatMessage[]>>(() =>
    storage.getJSON(KEYS.chats, {}),
  );
  const [chapterOutlines, setChapterOutlines] = useState<Record<string, string>>(() =>
    storage.getJSON(KEYS.chapters, {}),
  );

  // Persistence: derived from state so no handler can forget to write.
  useEffect(() => storage.setJSON(KEYS.chats, stripForStorage(chatHistories)), [chatHistories]);
  useEffect(() => storage.setJSON(KEYS.chapters, chapterOutlines), [chapterOutlines]);
  useEffect(() => storage.set(KEYS.activeVideo, activeVideoId), [activeVideoId]);
  useEffect(() => storage.set(KEYS.activePlaylist, activePlaylistId), [activePlaylistId]);
  useEffect(() => storage.set(KEYS.whisper, whisperModel), [whisperModel]);

  // Library fetch with a last-write-wins guard: a slow poll that resolves after
  // a newer one (or after a delete) must not overwrite fresher state.
  const fetchSeq = useRef(0);
  const fetchData = useCallback(async () => {
    const seq = ++fetchSeq.current;
    try {
      const [pData, vData] = await Promise.all([
        apiFetch<Playlist[]>('/api/playlists'),
        apiFetch<VideoMetadata[]>('/api/videos'),
      ]);
      if (seq !== fetchSeq.current) return;
      setPlaylists(pData);
      setAllVideos(vData);
      // Drop a restored selection that no longer exists in the library.
      setActiveVideoId((prev) => (prev && !vData.some((v) => v.video_id === prev) ? null : prev));
      setActivePlaylistId((prev) => (prev && !pData.some((p) => p.playlist_id === prev) ? null : prev));
    } catch (err) {
      console.error('Failed to load library data:', err);
    }
  }, []);

  useEffect(() => {
    // Initial library load. fetchData only sets state after its awaits
    // resolve, never synchronously, so the rule is a false positive here.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void fetchData();
  }, [fetchData]);

  // Poll while anything is processing/transcribing or being retried. The
  // interval is only recreated when that boolean flips, not on every tick.
  const anyBusy =
    playlists.some((pl) => pl.videos.some((v) => isBusyStatus(v.status))) ||
    allVideos.some((v) => isBusyStatus(v.status)) ||
    retryingIds.size > 0;

  useEffect(() => {
    if (!anyBusy) return;
    const id = window.setInterval(() => void fetchData(), POLL_MS);
    return () => window.clearInterval(id);
  }, [anyBusy, fetchData]);

  const parentPlaylistOf = useCallback(
    (videoId: string) => playlists.find((pl) => pl.videos.some((v) => v.video_id === videoId)) ?? null,
    [playlists],
  );

  const getChatHistory = (id: string, isPlaylist: boolean): ChatMessage[] =>
    chatHistories[chatKey(id, isPlaylist)] ?? welcome(isPlaylist);

  const handleMessagesChange = (id: string, isPlaylist: boolean, newMessages: ChatMessage[]) => {
    setChatHistories((prev) => ({ ...prev, [chatKey(id, isPlaylist)]: newMessages }));
  };

  const handleClearChat = (id: string, isPlaylist: boolean) => {
    if (!window.confirm('Are you sure you want to clear this chat history?')) return;
    setChatHistories((prev) => omitKeys(prev, [chatKey(id, isPlaylist)]));
  };

  const handleChaptersChange = (id: string, outlineMarkdown: string | null) => {
    setChapterOutlines((prev) =>
      outlineMarkdown === null ? omitKeys(prev, [id]) : { ...prev, [id]: outlineMarkdown },
    );
  };

  const handleSelectPlaylist = (playlistId: string) => {
    setActivePlaylistId(playlistId);
    setActiveVideoId(null);
  };

  const handleSelectVideo = (videoId: string) => {
    setActivePlaylistId(parentPlaylistOf(videoId)?.playlist_id ?? null);
    setActiveVideoId(videoId);
    setSeek(null);
  };

  const handleDeletePlaylist = async (playlistId: string) => {
    const nestedIds = playlists.find((p) => p.playlist_id === playlistId)?.videos.map((v) => v.video_id) ?? [];
    try {
      await apiFetch(`/api/playlists/${encodeURIComponent(playlistId)}`, { method: 'DELETE' });
    } catch (err) {
      alert(errorMessage(err, 'Failed to delete playlist.'));
      return;
    }
    setActivePlaylistId((prev) => (prev === playlistId ? null : prev));
    setActiveVideoId((prev) => (prev && nestedIds.includes(prev) ? null : prev));
    setChatHistories((prev) => omitKeys(prev, [chatKey(playlistId, true), ...nestedIds.map((id) => chatKey(id, false))]));
    setChapterOutlines((prev) => omitKeys(prev, nestedIds));
    void fetchData();
  };

  const handleDeleteVideo = async (videoId: string) => {
    try {
      await apiFetch(`/api/videos/${encodeURIComponent(videoId)}`, { method: 'DELETE' });
    } catch (err) {
      alert(errorMessage(err, 'Failed to delete video.'));
      return;
    }
    setActiveVideoId((prev) => (prev === videoId ? null : prev));
    setChatHistories((prev) => omitKeys(prev, [chatKey(videoId, false)]));
    setChapterOutlines((prev) => omitKeys(prev, [videoId]));
    void fetchData();
  };

  const handleRetryVideo = async (videoId: string) => {
    const vid = allVideos.find((v) => v.video_id === videoId) ?? playlists.flatMap((p) => p.videos).find((v) => v.video_id === videoId);
    if (!vid) return;
    setRetryingIds((prev) => new Set(prev).add(videoId));
    try {
      await apiFetch('/api/ingest', { method: 'POST', body: { url: vid.url, whisper_model: whisperModel } });
    } catch (err) {
      console.error('Retry failed:', errorMessage(err, 'unknown error'));
    }
    // Refresh before clearing the optimistic flag so the card does not flicker
    // back to "failed" for one round-trip.
    await fetchData();
    setRetryingIds((prev) => {
      const next = new Set(prev);
      next.delete(videoId);
      return next;
    });
  };

  const handleTimestampClick = (seconds: number, videoId?: string) => {
    if (videoId && videoId !== activeVideoId) {
      setActiveVideoId(videoId);
      setActivePlaylistId(parentPlaylistOf(videoId)?.playlist_id ?? null);
    }
    // A fresh nonce makes repeated clicks on the same timestamp seek again.
    setSeek({ seconds, nonce: ++seekNonce.current });
  };

  const handleGoHome = () => {
    setActiveVideoId(null);
    setActivePlaylistId(null);
  };

  const handleCloseWorkspace = () => {
    if (activeVideoId && activePlaylistId) {
      setActiveVideoId(null); // back to the folder dashboard
    } else {
      handleGoHome();
    }
  };

  // While a video is being retried, show it as 'processing' until the next fetch.
  const applyRetry = (v: VideoMetadata): VideoMetadata =>
    retryingIds.has(v.video_id) ? { ...v, status: 'processing' } : v;

  const playlistVideoIds = new Set(playlists.flatMap((pl) => pl.videos.map((v) => v.video_id)));
  const standaloneVideos = allVideos.filter((v) => !playlistVideoIds.has(v.video_id)).map(applyRetry);
  const displayPlaylists = playlists.map((pl) => ({ ...pl, videos: pl.videos.map(applyRetry) }));

  const activePlaylist = displayPlaylists.find((p) => p.playlist_id === activePlaylistId);
  const activeVideo = allVideos.find((v) => v.video_id === activeVideoId);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand" onClick={handleGoHome} title="Go to Home / Welcome Screen" style={{ cursor: 'pointer' }}>
          <GraduationCap className="brand-icon" size={24} />
          <h1>Lectura</h1>
          <span>RAG Teaching Assistant</span>
        </div>
        <KeyInput onKeyChange={setApiKey} />
      </header>

      <main className="main-layout">
        <Sidebar
          videos={standaloneVideos}
          playlists={displayPlaylists}
          whisperModel={whisperModel}
          onWhisperModelChange={setWhisperModel}
          activeVideoId={activeVideoId}
          activePlaylistId={activePlaylistId}
          onSelectVideo={handleSelectVideo}
          onSelectPlaylist={handleSelectPlaylist}
          onIngestSuccess={() => void fetchData()}
          onDeleteVideo={(id) => void handleDeleteVideo(id)}
          onDeletePlaylist={(id) => void handleDeletePlaylist(id)}
          onRetryVideo={(id) => void handleRetryVideo(id)}
        />

        <div className="content-area">
          {activeVideoId && activeVideo ? (
            <div className="workspace">
              <section className="player-section">
                <div className="player-header-bar">
                  {activePlaylistId ? (
                    <button className="workspace-back-btn" onClick={handleCloseWorkspace} title="Back to Playlist Folder">
                      <ArrowLeft size={13} />
                      <span>Back to Folder Dashboard</span>
                    </button>
                  ) : (
                    <button className="workspace-close-btn standalone-close" onClick={handleCloseWorkspace} title="Close Lecture Workspace / Go Home">
                      <X size={13} />
                      <span>Close Lecture</span>
                    </button>
                  )}
                </div>
                <VideoPlayer key={activeVideoId} videoId={activeVideoId} seek={seek} />
              </section>

              <div className="workspace-tabs-container">
                <div className="tab-headers">
                  <button className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <MessageSquare size={14} />
                      Lecture Q&A
                    </span>
                  </button>
                  <button className={`tab-btn ${activeTab === 'chapters' ? 'active' : ''}`} onClick={() => setActiveTab('chapters')}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <BookOpen size={14} />
                      Chapters & Summary Notes
                    </span>
                  </button>
                </div>

                <div className="tab-content">
                  <div className={`tab-pane ${activeTab === 'chat' ? 'active' : ''}`}>
                    <Chat
                      key={activeVideoId}
                      videoId={activeVideoId}
                      apiKey={apiKey}
                      onTimestampClick={handleTimestampClick}
                      messages={getChatHistory(activeVideoId, false)}
                      onMessagesChange={(newMessages) => handleMessagesChange(activeVideoId, false, newMessages)}
                      onClearChat={() => handleClearChat(activeVideoId, false)}
                    />
                  </div>
                  <div className={`tab-pane ${activeTab === 'chapters' ? 'active' : ''}`}>
                    <Chapters
                      key={activeVideoId}
                      videoId={activeVideoId}
                      videoTitle={activeVideo.title}
                      apiKey={apiKey}
                      onTimestampClick={handleTimestampClick}
                      chaptersMarkdown={chapterOutlines[activeVideoId] || null}
                      onChaptersChange={(markdown) => handleChaptersChange(activeVideoId, markdown)}
                    />
                  </div>
                </div>
              </div>
            </div>
          ) : activePlaylistId && activePlaylist ? (
            <div className="workspace">
              <section className="player-section">
                <div className="playlist-dashboard">
                  <div className="playlist-dashboard-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                      <div className="folder-icon-large">
                        <FolderOpen size={32} />
                      </div>
                      <div>
                        <h2>{activePlaylist.title}</h2>
                        <p className="playlist-meta-desc">
                          {activePlaylist.videos.length} Lecture{activePlaylist.videos.length !== 1 ? 's' : ''} • Playlist Folder
                        </p>
                      </div>
                    </div>
                    <button className="workspace-close-btn" onClick={handleCloseWorkspace} title="Close Playlist Workspace / Go Home">
                      <X size={14} />
                      <span>Close Folder</span>
                    </button>
                  </div>

                  <div className="playlist-grid">
                    {activePlaylist.videos.map((vid) => {
                      const isTranscribing = vid.status === 'transcribing';
                      const isBusy = isBusyStatus(vid.status);
                      const isFailed = vid.status === 'failed';
                      return (
                        <div
                          key={vid.video_id}
                          className={`playlist-grid-card ${isBusy ? 'processing' : ''} ${isFailed ? 'failed' : ''}`}
                          onClick={() => !isBusy && !isFailed && handleSelectVideo(vid.video_id)}
                        >
                          <div className="grid-thumb-wrapper">
                            <img src={vid.thumbnail} alt={vid.title} />
                            {!isBusy && !isFailed && (
                              <div className="play-overlay">
                                <Play size={24} />
                              </div>
                            )}
                            {isBusy && (
                              <div className="processing-overlay">
                                <Loader2 className="spinner" size={24} />
                                <span>{isTranscribing ? 'Transcribing locally…' : 'Processing…'}</span>
                              </div>
                            )}
                            {isFailed && (
                              <div className="failed-overlay">
                                <AlertCircle size={22} />
                                <span>Ingestion failed</span>
                                <button
                                  className="grid-retry-btn"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    void handleRetryVideo(vid.video_id);
                                  }}
                                  title={vid.error || 'Retry ingestion'}
                                >
                                  <RefreshCw size={14} /> Retry
                                </button>
                              </div>
                            )}
                          </div>
                          <div className="grid-card-info">
                            <h4 title={vid.title}>{vid.title}</h4>
                            <div className="grid-card-meta">
                              <span>{vid.author}</span>
                              {!isBusy && !isFailed && <span>{formatDuration(vid.duration)}</span>}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </section>

              <div className="workspace-tabs-container playlist-chat-container">
                <div className="tab-headers">
                  <div className="tab-btn active" style={{ cursor: 'default' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <MessageSquare size={14} />
                      Playlist Q&A Assistant
                    </span>
                  </div>
                </div>
                <div className="tab-content">
                  <div className="tab-pane active">
                    <Chat
                      key={activePlaylistId}
                      playlistId={activePlaylistId}
                      apiKey={apiKey}
                      onTimestampClick={handleTimestampClick}
                      messages={getChatHistory(activePlaylistId, true)}
                      onMessagesChange={(newMessages) => handleMessagesChange(activePlaylistId, true, newMessages)}
                      onClearChat={() => handleClearChat(activePlaylistId, true)}
                    />
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <Sparkles className="empty-icon" size={64} />
              <h2>Welcome to Lectura</h2>
              <p>
                To get started, paste a YouTube link of a lecture video or playlist in the left sidebar to index it.
                Or select an already processed lecture or playlist folder from the library!
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
