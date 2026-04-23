type Point = { x: number; y: number };

class DrawingApp {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private isDrawing: boolean = false;
  private lastPoint: Point | null = null;

  private penSizeInput: HTMLInputElement;
  private penSizeValue: HTMLElement;
  private penColorInput: HTMLInputElement;
  private eraserBtn: HTMLButtonElement;
  private clearBtn: HTMLButtonElement;
  private saveBtn: HTMLButtonElement;
  private loadBtn: HTMLButtonElement;

  private currentColor: string = "#000000";
  private isEraser: boolean = false;

  constructor(canvasId: string) {
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    this.ctx = this.canvas.getContext("2d")!;

    this.penSizeInput = document.getElementById("penSize") as HTMLInputElement;
    this.penSizeValue = document.getElementById("penSizeValue")!;
    this.penColorInput = document.getElementById("penColor") as HTMLInputElement;
    this.eraserBtn = document.getElementById("eraserBtn") as HTMLButtonElement;
    this.clearBtn = document.getElementById("clearBtn") as HTMLButtonElement;
    this.saveBtn = document.getElementById("saveBtn") as HTMLButtonElement;
    this.loadBtn = document.getElementById("loadBtn") as HTMLButtonElement;

    this.setupEventListeners();
    this.updatePenSizeDisplay();
    this.clearCanvas();
  }

  private setupEventListeners(): void {
    // マウスイベント
    this.canvas.addEventListener("mousedown", this.handleStart.bind(this));
    this.canvas.addEventListener("mousemove", this.handleMove.bind(this));
    this.canvas.addEventListener("mouseup", this.handleEnd.bind(this));
    this.canvas.addEventListener("mouseleave", this.handleEnd.bind(this));

    // タッチイベント（スマホ対応）
    this.canvas.addEventListener("touchstart", this.handleTouchStart.bind(this));
    this.canvas.addEventListener("touchmove", this.handleTouchMove.bind(this));
    this.canvas.addEventListener("touchend", this.handleTouchEnd.bind(this));

    // コントロール変更
    this.penSizeInput.addEventListener("input", () => {
      this.updatePenSizeDisplay();
    });

    this.penColorInput.addEventListener("input", () => {
      this.currentColor = this.penColorInput.value;
      this.isEraser = false;
      this.eraserBtn.textContent = "消しゴム";
    });

    this.eraserBtn.addEventListener("click", () => {
      this.isEraser = !this.isEraser;
      this.eraserBtn.textContent = this.isEraser ? "ペン" : "消しゴム";
    });

    this.clearBtn.addEventListener("click", () => {
      this.clearCanvas();
    });

    this.saveBtn.addEventListener("click", () => {
      this.saveDrawing();
    });

    this.loadBtn.addEventListener("click", () => {
      this.loadDrawing();
    });
  }

  private updatePenSizeDisplay(): void {
    this.penSizeValue.textContent = this.penSizeInput.value;
  }

  private getPenSize(): number {
    return parseInt(this.penSizeInput.value, 10);
  }

  private getCurrentColor(): string {
    return this.isEraser ? "#FFFFFF" : this.currentColor;
  }

  private clearCanvas(): void {
    this.ctx.fillStyle = "#FFFFFF";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // マウスイベントハンドラ
  private handleStart(e: MouseEvent): void {
    e.preventDefault();
    this.isDrawing = true;
    const point = this.getMousePoint(e);
    this.lastPoint = point;
    this.drawPoint(point);
  }

  private handleMove(e: MouseEvent): void {
    e.preventDefault();
    if (!this.isDrawing) return;

    const point = this.getMousePoint(e);
    this.drawLine(this.lastPoint!, point);
    this.lastPoint = point;
  }

  private handleEnd(e: MouseEvent): void {
    e.preventDefault();
    this.isDrawing = false;
    this.lastPoint = null;
  }

  // タッチイベントハンドラ
  private handleTouchStart(e: TouchEvent): void {
    e.preventDefault();
    this.isDrawing = true;
    const point = this.getTouchPoint(e);
    this.lastPoint = point;
    this.drawPoint(point);
  }

  private handleTouchMove(e: TouchEvent): void {
    e.preventDefault();
    if (!this.isDrawing) return;

    const point = this.getTouchPoint(e);
    this.drawLine(this.lastPoint!, point);
    this.lastPoint = point;
  }

  private handleTouchEnd(e: TouchEvent): void {
    e.preventDefault();
    this.isDrawing = false;
    this.lastPoint = null;
  }

  // 座標取得ユーティリティ
  private getMousePoint(e: MouseEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  private getTouchPoint(e: TouchEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    const touch = e.touches[0];
    return {
      x: touch.clientX - rect.left,
      y: touch.clientY - rect.top,
    };
  }

  // 描画ロジック
  private drawPoint(point: Point): void {
    this.ctx.beginPath();
    this.ctx.arc(point.x, point.y, this.getPenSize() / 2, 0, Math.PI * 2);
    this.ctx.fillStyle = this.getCurrentColor();
    this.ctx.fill();
  }

  private drawLine(start: Point, end: Point): void {
    this.ctx.beginPath();
    this.ctx.moveTo(start.x, start.y);
    this.ctx.lineTo(end.x, end.y);
    this.ctx.lineWidth = this.getPenSize();
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.strokeStyle = this.getCurrentColor();
    this.ctx.stroke();
  }

  // 保存・復元（簡易版：ローカルストレージにBase64で保存）
  private saveDrawing(): void {
    const dataUrl = this.canvas.toDataURL("image/png");
    localStorage.setItem("drawingAppData", dataUrl);
    alert("描画データを保存しました");
  }

  private loadDrawing(): void {
    const dataUrl = localStorage.getItem("drawingAppData");
    if (!dataUrl) {
      alert("保存されたデータがありません");
      return;
    }

    const img = new Image();
    img.onload = () => {
      this.clearCanvas();
      this.ctx.drawImage(img, 0, 0);
    };
    img.src = dataUrl;
  }
}

// アプリ起動
document.addEventListener("DOMContentLoaded", () => {
  new DrawingApp("canvas");
});