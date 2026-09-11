import React, { useState, useEffect, useRef } from 'react';
import { Eye, EyeOff, ShieldAlert, ShieldCheck } from 'lucide-react';
import { storage } from '../lib/storage';

const KEY_NAME = 'lectura_gemini_api_key';
const LEGACY_KEY_NAME = 'gemini_api_key';

const loadSavedKey = (): string => {
  const current = storage.get(KEY_NAME);
  if (current !== null) return current;
  // One-time migration from the un-namespaced key used by earlier builds.
  const legacy = storage.get(LEGACY_KEY_NAME);
  if (legacy) {
    storage.set(KEY_NAME, legacy);
    storage.set(LEGACY_KEY_NAME, null);
    return legacy;
  }
  return '';
};

interface KeyInputProps {
  onKeyChange: (key: string) => void;
}

export const KeyInput: React.FC<KeyInputProps> = ({ onKeyChange }) => {
  const [key, setKey] = useState<string>(loadSavedKey);
  const [showKey, setShowKey] = useState<boolean>(false);
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Informs the parent of the restored key on mount and of every change.
  // onKeyChange is a useState setter, so its identity is stable.
  useEffect(() => {
    onKeyChange(key);
  }, [key, onKeyChange]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setKey(val);
    storage.set(KEY_NAME, val);
  };

  const handleClear = () => {
    setKey('');
    storage.set(KEY_NAME, null);
    setIsOpen(false);
  };

  const isKeyEntered = key.trim().length > 0;

  return (
    <div className="api-key-section" ref={popoverRef}>
      <div
        className={`api-key-badge ${isKeyEntered ? 'active' : 'empty'}`}
        onClick={() => setIsOpen(!isOpen)}
        title="Click to configure Gemini API Key"
      >
        {isKeyEntered ? (
          <>
            <ShieldCheck size={14} />
            <span>API Key Active</span>
          </>
        ) : (
          <>
            <ShieldAlert size={14} />
            <span>Set API Key</span>
          </>
        )}
      </div>

      {isOpen && (
        <div className="api-key-popover">
          <h4>Gemini API Configuration</h4>
          <p>
            Enter your Google AI Studio API Key to authenticate queries. The key is stored locally in your browser's localStorage and sent only to your backend.
          </p>

          <div className="api-key-input-wrapper">
            <input
              type={showKey ? 'text' : 'password'}
              className="api-key-input"
              placeholder="Enter Google AI Studio API Key"
              value={key}
              onChange={handleKeyChange}
              style={{ width: '100%' }}
              autoFocus
            />
            <button
              type="button"
              className="api-key-toggle-btn"
              onClick={() => setShowKey(!showKey)}
              title={showKey ? 'Hide API Key' : 'Show API Key'}
            >
              {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>

          {isKeyEntered && (
            <button type="button" className="api-key-clear-btn" onClick={handleClear}>
              Clear Saved API Key
            </button>
          )}
        </div>
      )}
    </div>
  );
};
