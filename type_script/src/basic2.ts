// 基本的な型
let isDone: boolean = false;
let age: number = 25;
let userName: string = "たろう";

// 配列
let list: number[] = [1, 2, 3];
let fruits: Array<string> = ["apple", "banana"];

// どちらの型でもOKな場合 (Union型)
let id: number | string = 101;
id = "A101"; // OK


// 引数(x, y)と戻り値に型を指定
function add(x: number, y: number): number {
  return x + y;
}

// 戻り値がない場合は void を指定
function logMessage(message: string): void {
  console.log(message);
}

// アロー関数での記述
const multiply = (a: number, b: number): number => a * b;


interface User {
  id: number;
  name: string;
  email?: string; // 「?」をつけると省略可能（Optional）になる
}

const userA: User = {
  id: 1,
  name: "田中"
};

class Animal {
  // プロパティ
  private name: string;

  constructor(name: string) {
    this.name = name;
  }

  public move(distance: number): void {
    console.log(`${this.name} moved ${distance}m.`);
  }
}

const dog = new Animal("ポチ");
dog.move(10);


// T は使う時に決まる任意の型
function reverseList<T>(items: T[]): T[] {
  return items.reverse();
}

const numList = reverseList<number>([1, 2, 3]); // number型として動作
const strList = reverseList<string>(["a", "b", "c"]); // string型として動作