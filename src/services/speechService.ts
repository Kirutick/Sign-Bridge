/**
 * Web Speech API Text-to-Speech service for Indian Sign Language English sentence output.
 * Strictly uses English speech synthesis only:
 * - Prioritizes Indian English (en-IN) where available.
 * - Falls back to en-US or general English voices.
 * - Never selects or exposes Hindi (hi-IN) or non-English voices.
 */

export interface SpeechOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
  voiceURI?: string;
  lang?: string;
}

export class SpeechService {
  private currentUtterance: SpeechSynthesisUtterance | null = null;
  private voices: SpeechSynthesisVoice[] = [];
  private voicesLoaded: boolean = false;

  constructor() {
    if (this.isSupported()) {
      this.initVoices();
    }
  }

  public isSupported(): boolean {
    return typeof window !== "undefined" && "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
  }

  private initVoices(): void {
    if (!this.isSupported()) return;

    const populate = () => {
      try {
        const rawVoices = window.speechSynthesis.getVoices() || [];
        // Strict invariant: Expose English voices only (en-IN, en-US, en-GB, etc.)
        this.voices = rawVoices.filter((v) => v.lang.toLowerCase().startsWith("en"));
        this.voicesLoaded = this.voices.length > 0;
      } catch (e) {
        console.warn("Could not retrieve speechSynthesis voices:", e);
      }
    };

    populate();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = populate;
    }
  }

  /**
   * Returns available speech synthesis voices, strictly filtered to English.
   */
  public getVoices(): SpeechSynthesisVoice[] {
    if (!this.voicesLoaded && this.isSupported()) {
      const rawVoices = window.speechSynthesis.getVoices() || [];
      this.voices = rawVoices.filter((v) => v.lang.toLowerCase().startsWith("en"));
      this.voicesLoaded = this.voices.length > 0;
    }
    return this.voices;
  }

  public isSpeaking(): boolean {
    if (!this.isSupported()) return false;
    return window.speechSynthesis.speaking || this.currentUtterance !== null;
  }

  public stop(): void {
    if (!this.isSupported()) return;
    try {
      window.speechSynthesis.cancel();
      this.currentUtterance = null;
    } catch (e) {
      console.warn("Error stopping speech synthesis:", e);
    }
  }

  public speak(text: string, options: SpeechOptions = {}): Promise<void> {
    return new Promise((resolve) => {
      if (!this.isSupported()) {
        console.warn("Web Speech API is not supported in this environment.");
        resolve();
        return;
      }

      if (!text || text.trim().length === 0) {
        resolve();
        return;
      }

      this.stop(); // Stop any currently playing speech

      const utterance = new SpeechSynthesisUtterance(text.trim());
      utterance.rate = options.rate ?? 1.0;
      utterance.pitch = options.pitch ?? 1.0;
      utterance.volume = options.volume ?? 1.0;

      // Select voice: Prioritize en-IN, then en-US, then any English voice
      const englishVoices = this.getVoices();
      if (options.voiceURI) {
        const matchingVoice = englishVoices.find((v) => v.voiceURI === options.voiceURI);
        if (matchingVoice) utterance.voice = matchingVoice;
      } else if (englishVoices.length > 0) {
        const indianEnglishVoice = englishVoices.find((v) => v.lang.toLowerCase().includes("en-in"));
        if (indianEnglishVoice) {
          utterance.voice = indianEnglishVoice;
        } else {
          const usEnglishVoice = englishVoices.find((v) => v.lang.toLowerCase().includes("en-us"));
          utterance.voice = usEnglishVoice || englishVoices[0];
        }
      }

      // Default language to English
      utterance.lang = options.lang || "en-IN";

      utterance.onend = () => {
        this.currentUtterance = null;
        resolve();
      };

      utterance.onerror = (e) => {
        console.warn("Speech synthesis utterance error:", e);
        this.currentUtterance = null;
        resolve();
      };

      this.currentUtterance = utterance;
      window.speechSynthesis.speak(utterance);
    });
  }
}

export const speechService = new SpeechService();
