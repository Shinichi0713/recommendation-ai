// ボタンクリックでカウントアップするシンプルなカウンター
class Counter {
  private count: number = 0;
  private displayElement: HTMLElement;
  private buttonElement: HTMLButtonElement;

  constructor(displayId: string, buttonId: string) {
    this.displayElement = document.getElementById(displayId)!;
    this.buttonElement = document.getElementById(buttonId)! as HTMLButtonElement;

    this.buttonElement.addEventListener("click", () => {
      this.increment();
    });

    this.updateDisplay();
  }

  private increment(): void {
    this.count++;
    this.updateDisplay();
  }

  private updateDisplay(): void {
    this.displayElement.textContent = `Count: ${this.count}`;
  }
}

// HTMLが読み込まれたらカウンターを初期化
document.addEventListener("DOMContentLoaded", () => {
  new Counter("counter-display", "increment-button");
});