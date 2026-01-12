// 1. 取引の種類を定義する列挙型
enum TransactionType {
    Deposit = "入金",
    Withdraw = "出金"
}

// 2. ユーザーの形を定義するインターフェース
interface User {
    id: number;
    name: string;
}

// 3. 銀行口座クラス
class BankAccount {
    private balance: number; // 外部から直接書き換えられないようにprivateに
    public owner: User;

    constructor(owner: User, initialBalance: number) {
        this.owner = owner;
        this.balance = initialBalance;
    }

    // 入金メソッド
    deposit(amount: number): void {
        this.balance += amount;
        this.logTransaction(TransactionType.Deposit, amount);
    }

    // 出金メソッド
    withdraw(amount: number): boolean {
        if (amount > this.balance) {
            console.log(`[エラー] 残高不足です。現在の残高: ${this.balance}円`);
            return false;
        }
        this.balance -= amount;
        this.logTransaction(TransactionType.Withdraw, amount);
        return true;
    }

    // 現在の残高を表示（読み取り専用プロパティのように振る舞う）
    getBalance(): number {
        return this.balance;
    }

    // 内部用のログ出力
    private logTransaction(type: TransactionType, amount: number): void {
        console.log(`${this.owner.name}さんの口座に ${amount}円 の${type}がありました。`);
        console.log(`現在の残高: ${this.balance}円`);
    }
}

// --- 実行コード ---

const myAccount = new BankAccount({ id: 1, name: "田中太郎" }, 5000);

myAccount.deposit(3000);  // 3000円入金
myAccount.withdraw(2000); // 2000円出金
myAccount.withdraw(10000); // 残高不足エラー

console.log(`最終的な ${myAccount.owner.name} さんの残高は ${myAccount.getBalance()}円 です。`);