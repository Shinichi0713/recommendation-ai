export {}; // これを書くと「このファイルは独立したモジュールである」と認識される

const name: string = "Taro";
console.log(name);

let isDone: boolean = false;
let count: number = 10;
let name: string = "Gemini";

// 配列
let list: number[] = [1, 2, 3];
let fruits: Array<string> = ["apple", "banana"];

// 何でも許容する（多用は避けるべき）
let anything: any = 4;


function add(x: number, y: number): number {
  return x + y;
}

// 戻り値がない場合は void
function logMessage(message: string): void {
  console.log(message);
}


// Interfaceの例
interface User {
  id: number;
  name: string;
  age?: number; // 「?」をつけると、あってもなくても良い（任意項目）になる
}

const userA: User = { id: 1, name: "Tanaka" };

// Type Aliasの例（より柔軟な定義が可能）
type ID = number | string; // Union型：数値か文字列どちらか
const myId: ID = "A100";


enum Direction {
  Up,    // 0
  Down,  // 1
  Left,  // 2
  Right, // 3
}

let move: Direction = Direction.Up;


// 渡された型と同じ型の配列を返す関数
function identity<T>(arg: T): T {
  return arg;
}

let output = identity<string>("hello");