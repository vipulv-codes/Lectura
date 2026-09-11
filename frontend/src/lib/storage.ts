// localStorage can throw (disabled storage, private mode, quota). Every access
// goes through here so a storage failure never takes down a UI handler.

export const storage = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string | null): void {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch (e) {
      console.warn(`localStorage write failed for ${key}:`, e);
    }
  },
  getJSON<T>(key: string, fallback: T): T {
    const raw = storage.get(key);
    if (raw === null) return fallback;
    try {
      return JSON.parse(raw) as T;
    } catch {
      return fallback;
    }
  },
  setJSON(key: string, value: unknown): void {
    storage.set(key, JSON.stringify(value));
  },
};
