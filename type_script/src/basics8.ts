// ユーザー情報の型を定義
interface User {
  id: number;
  name: string;
  email: string;
  age?: number; // 省略可能なプロパティ
}

// ユーザー配列を管理するクラス
class UserManager {
  private users: User[] = [];

  addUser(user: User): void {
    this.users.push(user);
  }

  findUserById(id: number): User | undefined {
    return this.users.find(user => user.id === id);
  }

  getUsersOverAge(age: number): User[] {
    return this.users.filter(user => user.age != null && user.age > age);
  }
}

// 使用例
const manager = new UserManager();

manager.addUser({
  id: 1,
  name: "山田 太郎",
  email: "taro@example.com",
  age: 30,
});

manager.addUser({
  id: 2,
  name: "佐藤 花子",
  email: "hanako@example.com",
  // age は省略可能なので指定しない
});

const user1 = manager.findUserById(1);
console.log(user1?.name); // "山田 太郎"

const adults = manager.getUsersOverAge(20);
console.log(adults.length); // 1（ageが指定されているユーザーのみ）


// ジェネリクスを使ったシンプルなストア
class SimpleStore<T> {
  private data: Map<string, T> = new Map();

  set(key: string, value: T): void {
    this.data.set(key, value);
  }

  get(key: string): T | undefined {
    return this.data.get(key);
  }

  has(key: string): boolean {
    return this.data.has(key);
  }
}

// 使用例
const userStore = new SimpleStore<User>();

userStore.set("user-1", {
  id: 1,
  name: "鈴木 一郎",
  email: "ichiro@example.com",
});

const u = userStore.get("user-1");
console.log(u?.name); // "鈴木 一郎"