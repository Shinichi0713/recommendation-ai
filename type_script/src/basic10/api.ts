import { User } from "./store";

// モックデータ
const mockUsers: User[] = [
  { id: 1, name: "山田 太郎", email: "taro@example.com", age: 30 },
  { id: 2, name: "佐藤 花子", email: "hanako@example.com", age: 25 },
  { id: 3, name: "鈴木 一郎", email: "ichiro@example.com" },
];

// 非同期でユーザー一覧を取得
export async function fetchUsers(): Promise<User[]> {
  await new Promise(resolve => setTimeout(resolve, 500)); // 擬似的な遅延
  return [...mockUsers];
}

// IDでユーザーを取得
export async function fetchUserById(id: number): Promise<User | null> {
  await new Promise(resolve => setTimeout(resolve, 300));
  return mockUsers.find(user => user.id === id) ?? null;
}