// 変数に型を指定
let username: string = "Gemini";
let age: number = 25;
let isStudent: boolean = false;

// 関数の引数と戻り値に型を指定
// (引数aは数値, 引数bは数値, 戻り値も数値)
function add(a: number, b: number): number {
  return a + b;
}

console.log(add(10, 5)); // 15
// add("10", 5); // 型が違うとエディタがエラーを出してくれる


// ユーザー情報の構造を定義
interface User {
  id: number;
  name: string;
  email: string;
  age?: number; // 「?」をつけると、あってもなくても良い（任意）という意味になる
}

const newUser: User = {
  id: 1,
  name: "田中太郎",
  email: "tanaka@example.com"
  // age はなくてもエラーにならない
};

function greet(user: User) {
  console.log(`こんにちは、${user.name}さん！`);
}

greet(newUser);

class Animal {
  // privateを付けるとクラスの外からは読み書きできない
  private species: string;

  constructor(species: string) {
    this.species = species;
  }

  public announce() {
    console.log(`この動物は ${this.species} です。`);
  }
}

const dog = new Animal("イヌ");
dog.announce(); // 実行可能
// console.log(dog.species); // Error: privateなので外からはアクセス不可


// Union型（数値か文字列、どちらか）
let id: number | string;
id = 101;
id = "A101";

// Literal型（決まった値しか入れられない）
let status: "success" | "error" | "loading";
status = "success"; 
// status = "pending"; // エラーになる