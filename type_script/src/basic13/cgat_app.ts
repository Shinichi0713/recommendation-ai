// メッセージの型定義
interface Message {
  id: number;
  sender: string;
  text: string;
  timestamp: Date;
  isOwn: boolean;
}

class ChatApp {
  private messages: Message[] = [];
  private nextId: number = 1;
  private currentUser: string = "匿名";

  private userNameInput: HTMLInputElement;
  private setNameBtn: HTMLButtonElement;
  private chatArea: HTMLDivElement;
  private messageInput: HTMLInputElement;
  private sendBtn: HTMLButtonElement;

  constructor() {
    this.userNameInput = document.getElementById("userNameInput") as HTMLInputElement;
    this.setNameBtn = document.getElementById("setNameBtn") as HTMLButtonElement;
    this.chatArea = document.getElementById("chatArea") as HTMLDivElement;
    this.messageInput = document.getElementById("messageInput") as HTMLInputElement;
    this.sendBtn = document.getElementById("sendBtn") as HTMLButtonElement;

    this.setupEventListeners();
    this.loadMessages();
    this.render();
  }

  private setupEventListeners(): void {
    // ユーザー名設定
    this.setNameBtn.addEventListener("click", () => {
      const name = this.userNameInput.value.trim() || "匿名";
      this.currentUser = name;
      alert(`ユーザー名を「${name}」に設定しました`);
    });

    // メッセージ送信
    this.sendBtn.addEventListener("click", () => this.sendMessage());
    this.messageInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        this.sendMessage();
      }
    });
  }

  private sendMessage(): void {
    const text = this.messageInput.value.trim();
    if (text === "") return;

    const message: Message = {
      id: this.nextId++,
      sender: this.currentUser,
      text,
      timestamp: new Date(),
      isOwn: true,
    };

    this.messages.push(message);
    this.messageInput.value = "";
    this.saveMessages();
    this.render();

    // モック：少し遅れて「相手」からの返信をシミュレート
    this.simulateReply(text);
  }

  private simulateReply(userMessage: string): void {
    setTimeout(() => {
      const botMessages = [
        "そうなんだ！",
        "なるほど〜",
        "それで？",
        "面白いね！",
        "もっと教えて！",
        "わかったよ！",
      ];
      const randomReply = botMessages[Math.floor(Math.random() * botMessages.length)];

      const reply: Message = {
        id: this.nextId++,
        sender: "Bot",
        text: randomReply,
        timestamp: new Date(),
        isOwn: false,
      };

      this.messages.push(reply);
      this.saveMessages();
      this.render();
    }, 1000 + Math.random() * 2000); // 1〜3秒後に返信
  }

  private render(): void {
    this.chatArea.innerHTML = "";

    if (this.messages.length === 0) {
      const emptyMsg = document.createElement("div");
      emptyMsg.textContent = "メッセージがありません。最初のメッセージを送ってみましょう！";
      emptyMsg.style.textAlign = "center";
      emptyMsg.style.color = "#999";
      emptyMsg.style.padding = "20px";
      this.chatArea.appendChild(emptyMsg);
      return;
    }

    this.messages.forEach(msg => {
      const msgDiv = document.createElement("div");
      msgDiv.className = `message ${msg.isOwn ? "own" : "other"}`;

      const senderSpan = document.createElement("div");
      senderSpan.className = "sender";
      senderSpan.textContent = msg.sender;

      const textSpan = document.createElement("div");
      textSpan.textContent = msg.text;

      const timeSpan = document.createElement("div");
      timeSpan.className = "sender";
      timeSpan.textContent = this.formatTime(msg.timestamp);

      msgDiv.appendChild(senderSpan);
      msgDiv.appendChild(textSpan);
      msgDiv.appendChild(timeSpan);

      this.chatArea.appendChild(msgDiv);
    });

    // 最新メッセージまでスクロール
    this.chatArea.scrollTop = this.chatArea.scrollHeight;
  }

  private formatTime(date: Date): string {
    return `${date.getHours().toString().padStart(2, "0")}:${date.getMinutes().toString().padStart(2, "0")}`;
  }

  // ローカルストレージへの保存・復元
  private saveMessages(): void {
    const data = JSON.stringify({
      messages: this.messages,
      nextId: this.nextId,
    });
    localStorage.setItem("chatAppData", data);
  }

  private loadMessages(): void {
    const data = localStorage.getItem("chatAppData");
    if (!data) return;

    try {
      const parsed = JSON.parse(data);
      this.messages = (parsed.messages || []).map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
      }));
      this.nextId = parsed.nextId || 1;
    } catch (err) {
      console.error("メッセージデータの読み込みに失敗しました", err);
    }
  }
}

// アプリ起動
document.addEventListener("DOMContentLoaded", () => {
  new ChatApp();
});