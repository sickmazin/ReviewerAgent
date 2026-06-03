// API client for the Reviewer Agent backend (FastAPI).
// Always connects to the backend; no mock fallback.

export type ScoreLabel = "BAD" | "GOOD" | "EXCELLENT" | "ERROR";

export interface Chat {
  id: string; // UUID
  title: string;
  created_at: string;
}

export interface Issue {
  start: number;
  end: number;
  type: "error" | "improve" | "suggestion";
  message: string;
}

export interface HighlightsData {
  text: string;
  issues: Issue[];
}

export interface ReviewDetails {
  word_count?: number;
  char_count?: number;
}

export interface Review {
  id: number;
  chat_id: string; // UUID
  text: string;
  site: string;
  url: string | null;
  score: number | string | null;
  is_generic_compliant: boolean | null;
  follow_guidelines: boolean | null;
  grammar_errors: boolean | null;
  title: string | null;
  reasoning: string | null;
  highlights: HighlightsData | null;
  details: ReviewDetails | null;
  created_at: string;
}

export interface Site {
  id: string;
  label: string;
  icon?: string;
}

export interface ScoreCategory {
  label: string;
  description: string;
}

export interface ModelInfo {
  welcome_title: string;
  welcome_description: string;
  how_it_works: string[];
  model_details: { label: string; value: string }[];
  score_categories: ScoreCategory[];
  score_dimensions: { label: string; description: string }[];
  metrics_description: string;
}

export interface EvaluateRequest {
  chat_id: string; // "0" for new chat or UUID string
  text: string;
  category: string;
  rating?: number;
  model?: string;
}

const BASE_URL: string = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

// ---------- PUBLIC API ----------

export const api = {
  async getChats(): Promise<Chat[]> {
    return http<Chat[]>("/chats");
  },
  async deleteChat(id: string): Promise<void> {
    await http<void>(`/chats/${id}`, { method: "DELETE" });
  },
  async getChatReviews(chat_id: string): Promise<Review[]> {
    return http<Review[]>(`/chats/${chat_id}/review`);
  },
  async getSites(): Promise<Site[]> {
    return http<Site[]>("/sites");
  },
  async getModelInfo(): Promise<ModelInfo> {
    return http<ModelInfo>("/model-info");
  },
  async evaluate(req: EvaluateRequest): Promise<Review> {
    return http<Review>("/evaluate", { method: "POST", body: JSON.stringify(req) });
  },
};
