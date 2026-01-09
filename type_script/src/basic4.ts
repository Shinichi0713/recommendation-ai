// 基本的な型
letの名前: string = "Gemini";
let 年齢: number = 25;
let 正解か: boolean = true;

// 配列の定義（2種類の書き方があります）
let 数値配列: number[] = [1, 2, 3];
let 文字列配列: Array<string> = ["Apple", "Banana"];

// タプル（要素数と型が固定された配列）
let ユーザー: [number, string] = [1, "田中"];

// 引数がnumber型、戻り値がnumber型の関数
function 合計(a: number, b: number): number {
  return a + b;
}

// 戻り値がない場合は「void」を指定
function 挨拶(name: string): void {
  console.log("こんにちは、" + name + "さん");
}

// アロー関数の場合
const 掛け算 = (x: number, y: number): number => x * y;

// Interface（インターフェース）: オブジェクトの形を定義するのに一般的
interface User {
  id: number;
  name: string;
  email?: string; // 「?」をつけると、あってもなくても良い（任意）になる
}

const customer: User = {
  id: 101,
  name: "佐藤"
  // emailは省略可能
};

// Type Alias（型エイリアス）: より柔軟な型定義
type Point = {
  x: number;
  y: number;
};

// 数値か文字列、どちらかを受け取れる
let 答え: number | string;

答え = 42;      // OK
答え = "不明";  // OK
// 答え = true; // エラー（booleanは許可されていない）


enum Status {
  Start,   // 0
  Process, // 1
  End      // 2
}

let currentStatus: Status = Status.Start;


// <T> は、後で決まる型（Type）の略
function 配列の先頭を取得<T>(array: T[]): T {
  return array[0];
}

const n = 配列の先頭を取得<number>([10, 20, 30]); // nはnumber型になる
const s = 配列の先頭を取得<string>(["A", "B", "C"]); // sはstring型になる

// 設計図（インターフェース）の定義
interface User {
  id: number;
  name: string;
  age: number;
}

// 設計図に基づいたオブジェクトの作成
const user1: User = {
  id: 1,
  name: "田中",
  age: 30
};

// 【エラー例】
const user2: User = {
  id: 2,
  name: "佐藤"
  // age がないので、TypeScriptが「設計図通りじゃないですよ！」と警告を出す
};


interface User {
  id: number;
  name: string;
  email?: string; // emailはあってもなくてもOK
}

interface Config {
  readonly apiKey: string;
}

const myConfig: Config = { apiKey: "12345" };
// myConfig.apiKey = "67890"; // エラー！書き換え不可

interface Animal {
  name: string;
}

// Animalを継承してDogを作る
interface Dog extends Animal {
  breed: string; // 犬種を追加
}

const myDog: Dog = {
  name: "ポチ",
  breed: "柴犬"
};

