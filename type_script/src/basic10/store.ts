// ユーザー情報の型
export interface User {
  id: number;
  name: string;
  email: string;
  age?: number;
}

// アプリ全体の状態
export interface AppState {
  users: User[];
  currentUser: User | null;
  loading: boolean;
  error: string | null;
}

// 状態更新用のアクション
export type Action =
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_USERS"; payload: User[] }
  | { type: "SET_CURRENT_USER"; payload: User | null };

// シンプルなストア（Redux風）
export class Store {
  private state: AppState;
  private listeners: Array<(state: AppState) => void> = [];

  constructor(initialState: AppState) {
    this.state = initialState;
  }

  getState(): AppState {
    return this.state;
  }

  dispatch(action: Action): void {
    this.state = this.reducer(this.state, action);
    this.listeners.forEach(listener => listener(this.state));
  }

  subscribe(listener: (state: AppState) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private reducer(state: AppState, action: Action): AppState {
    switch (action.type) {
      case "SET_LOADING":
        return { ...state, loading: action.payload };
      case "SET_ERROR":
        return { ...state, error: action.payload };
      case "SET_USERS":
        return { ...state, users: action.payload };
      case "SET_CURRENT_USER":
        return { ...state, currentUser: action.payload };
      default:
        return state;
    }
  }
}

// 初期状態
export const initialState: AppState = {
  users: [],
  currentUser: null,
  loading: false,
  error: null,
};

export const store = new Store(initialState);