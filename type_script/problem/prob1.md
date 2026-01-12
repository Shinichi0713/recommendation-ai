承知いたしました。実際に手を動かしてコードを書く、実践的な3つのステップの例題を用意しました。

TypeScriptの特徴である **「インターフェース」「Union型」「ジェネリクス」** を活用した問題です。

---

### 第1問：インターフェースと関数の型定義

【お題】

ショッピングサイトの「商品（Product）」を管理するプログラムを作成してください。

1. `Product` という名前のインターフェースを作成してください。
   * `id`: 数値型
   * `name`: 文字列型
   * `price`: 数値型
   * `description`: 文字列型（ただし、**省略可能**とする）
2. `Product` の配列を受け取り、価格が 1000円以上の商品だけを絞り込んで返す関数 `filterExpensiveProducts` を作成してください。

---

### 第2問：Union型とTypeガード

【お題】

「管理者」と「一般ユーザー」で異なるメッセージを表示する関数を作成してください。

1. 以下の2つの型を作成してください。
   * `Admin`: `role` が `"admin"` (文字列リテラル型)、`editPermission` が `boolean`
   * `User`: `role` が `"user"` (文字列リテラル型)、`id` が `number`
2. 引数に `Admin | User` を受け取る関数 `checkAccess` を作成してください。
3. 関数の中で `if` 文を使い、`role` が `"admin"` の場合は「管理画面へようこそ」、それ以外の場合は「ユーザー画面へようこそ」と `console.log` で出力してください。

---

### 第3問：ジェネリクス（Generics）

【お題】

どんな型でも使える「APIレスポンスの形」を定義してください。

1. ジェネリクス型 `ApiResponse<T>` を作成してください。プロパティは以下の通りです。
   * `status`: `"success"` または `"error"`
   * `data`: 型 `T`（ここに実際のデータが入る）
2. この `ApiResponse` 型を使って、以下の2つの変数を作成してください。
   * 文字列の配列が入る成功レスポンス（例：`["Apple", "Banana"]`）
   * 数値が入る成功レスポンス（例：`100`）

---

### ヒント

困ったときは、以下のコード構造を参考にしてください。

**TypeScript**

```
// 第1問のヒント
interface Product { ... }
const filterExpensiveProducts = (items: Product[]): Product[] => {
  return items.filter(item => item.price >= 1000);
};

// 第2問のヒント
type Admin = { role: "admin"; ... };
type User = { role: "user"; ... };

// 第3問のヒント
interface ApiResponse<T> {
  data: T;
  ...
}
```

これらの問題の解答例を確認したいですか？それとも、ご自身で書いたコードの添削を希望されますか？
