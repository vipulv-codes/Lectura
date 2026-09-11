export const formatDuration = (secs: number): string => {
  const total = Math.max(0, Math.floor(secs || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => (n < 10 ? `0${n}` : `${n}`);
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
};

export type VideoStatus = 'processing' | 'transcribing' | 'indexed' | 'failed';

export const isBusyStatus = (status?: VideoStatus): boolean =>
  status === undefined || status === 'processing' || status === 'transcribing';
