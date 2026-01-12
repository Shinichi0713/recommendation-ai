// 1. Productインターフェースの定義
interface Product {
  id: number;
  name: string;
  price: number;
  description?: string; // 「?」で省略可能（Optional）に
}

// 2. 1000円以上の商品を絞り込む関数
// 引数に Product[] (配列)、戻り値も Product[] と定義します
function filterExpensiveProducts(items: Product[]): Product[] {
  return items.filter((item) => item.price >= 1000);
}

// --- 動作確認 ---

const sampleProducts: Product[] = [
  { id: 1, name: "消しゴム", price: 100 },
  { id: 2, name: "高級ボールペン", price: 1200, description: "書き心地抜群です" },
  { id: 3, name: "ノート", price: 200 },
  { id: 4, name: "レザーペンケース", price: 3500 } // descriptionを省略
];

const expensiveItems = filterExpensiveProducts(sampleProducts);

console.log("1000円以上の商品一覧:");
expensiveItems.forEach((item) => {
  console.log(`${item.name}: ${item.price}円`);
});