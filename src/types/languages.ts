export type SignLanguageKey = "isl";

export interface SignLanguageMetadata {
  key: SignLanguageKey;
  name: string;
  signLanguage: string;
  outputLanguage: string;
  flagEmoji: string;
  region: string;
  description: string;
  modelDir: string;
  vocabularyCount: number;
  featuredVocabulary: string[];
}

export const SIGN_LANGUAGES: Record<SignLanguageKey, SignLanguageMetadata> = {
  isl: {
    key: "isl",
    name: "Indian Sign Language",
    signLanguage: "Indian Sign Language (ISL)",
    outputLanguage: "English",
    flagEmoji: "🇮🇳",
    region: "India",
    description: "Indian Sign Language (ISL) temporal recognition with English text and speech output.",
    modelDir: "/models/sign-model-isl",
    vocabularyCount: 50,
    featuredVocabulary: ["HELLO", "THANK_YOU", "YES", "NO", "HELP", "PLEASE", "SORRY", "GOODBYE", "WATER", "FOOD", "STOP", "GO", "AND", "INDIA"],
  },
};
