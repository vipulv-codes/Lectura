import React, { useEffect, useRef } from 'react';

// Minimal typings for the subset of the YouTube IFrame Player API we use.
interface YouTubePlayer {
  seekTo: (seconds: number, allowSeekAhead: boolean) => void;
  playVideo: () => void;
  destroy: () => void;
}

interface YouTubePlayerEvent {
  target: YouTubePlayer;
}

interface YouTubeNamespace {
  Player: new (elementId: string, options: Record<string, unknown>) => YouTubePlayer;
}

declare global {
  interface Window {
    YT?: YouTubeNamespace;
    onYouTubeIframeAPIReady?: () => void;
  }
}

// A seek is a one-shot command, not derived state: the nonce lets the same
// timestamp be requested twice in a row.
export type SeekRequest = { seconds: number; nonce: number } | null;

interface VideoPlayerProps {
  videoId: string;
  seek: SeekRequest;
}

const API_SRC = 'https://www.youtube.com/iframe_api';

const ensureApiScript = () => {
  if (window.YT?.Player) return;
  if (document.querySelector(`script[src="${API_SRC}"]`)) return;
  const tag = document.createElement('script');
  tag.src = API_SRC;
  document.head.appendChild(tag);
};

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoId, seek }) => {
  const playerRef = useRef<YouTubePlayer | null>(null);
  // The seek to apply once the player is ready (initial seek for a video
  // opened via a citation from another lecture).
  const pendingSeek = useRef<SeekRequest>(seek);
  const containerId = `yt-player-${videoId}`;

  useEffect(() => {
    let cancelled = false;
    ensureApiScript();

    const initPlayer = () => {
      if (cancelled || playerRef.current || !window.YT?.Player) return;
      try {
        new window.YT.Player(containerId, {
          height: '100%',
          width: '100%',
          videoId,
          playerVars: { autoplay: 0, modestbranding: 1, rel: 0, origin: window.location.origin },
          events: {
            onReady: (event: YouTubePlayerEvent) => {
              if (cancelled) {
                event.target.destroy();
                return;
              }
              playerRef.current = event.target;
              const initial = pendingSeek.current;
              if (initial && initial.seconds >= 0) {
                event.target.seekTo(initial.seconds, true);
                event.target.playVideo();
              }
            },
          },
        });
      } catch (err) {
        console.error('Failed to initialize YouTube Player:', err);
      }
    };

    let previousCallback: (() => void) | undefined;
    if (window.YT?.Player) {
      initPlayer();
    } else {
      previousCallback = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => {
        previousCallback?.();
        initPlayer();
      };
    }

    return () => {
      cancelled = true;
      // Restore whatever callback was registered before us so the chain does
      // not grow by one closure per mount.
      if (previousCallback !== undefined || window.onYouTubeIframeAPIReady) {
        window.onYouTubeIframeAPIReady = previousCallback;
      }
      playerRef.current?.destroy();
      playerRef.current = null;
    };
    // containerId derives from videoId; the initial seek is read from a ref.
  }, [videoId, containerId]);

  useEffect(() => {
    if (!seek || seek.seconds < 0) return;
    const player = playerRef.current;
    if (player) {
      player.seekTo(seek.seconds, true);
      player.playVideo();
    } else {
      pendingSeek.current = seek;
    }
  }, [seek]);

  return (
    <div className="player-wrapper">
      <div style={{ width: '100%', height: '100%', aspectRatio: '16/9' }}>
        <div id={containerId} style={{ width: '100%', height: '100%' }} />
      </div>
    </div>
  );
};
