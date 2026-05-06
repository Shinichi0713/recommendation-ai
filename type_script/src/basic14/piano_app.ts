// 音階の定義（C4〜B4の1オクターブ）
const NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;
type NoteName = typeof NOTES[number];

// 周波数テーブル（C4 = 261.63Hz）
const FREQUENCIES: Record<NoteName, number> = {
  "C": 261.63,
  "C#": 277.18,
  "D": 293.66,
  "D#": 311.13,
  "E": 329.63,
  "F": 349.23,
  "F#": 369.99,
  "G": 392.00,
  "G#": 415.30,
  "A": 440.00,
  "A#": 466.16,
  "B": 493.88,
};

// 白鍵のみの並び（C, D, E, F, G, A, B）
const WHITE_NOTES: NoteName[] = ["C", "D", "E", "F", "G", "A", "B"];

// 黒鍵のみの並び（C#, D#, F#, G#, A#）
const BLACK_NOTES: NoteName[] = ["C#", "D#", "F#", "G#", "A#"];

// キーボードショートカットのマッピング
const KEY_MAP: Record<string, NoteName> = {
  "a": "C",
  "s": "D",
  "d": "E",
  "f": "F",
  "g": "G",
  "h": "A",
  "j": "B",
  "w": "C#",
  "e": "D#",
  "t": "F#",
  "y": "G#",
  "u": "A#",
};

class PianoApp {
  private piano: HTMLElement;
  private whiteKeysContainer: HTMLElement;
  private blackKeysContainer: HTMLElement;
  private audioContext: AudioContext | null = null;
  private activeOscillators: Map<NoteName, OscillatorNode> = new Map();

  constructor(pianoId: string) {
    this.piano = document.getElementById(pianoId)!;
    this.whiteKeysContainer = document.getElementById("whiteKeys")!;
    this.blackKeysContainer = document.getElementById("blackKeys")!;

    this.setupAudio();
    this.renderKeys();
    this.setupEventListeners();
  }

  private setupAudio(): void {
    // ユーザー操作（クリックなど）でAudioContextを開始
    this.piano.addEventListener("mousedown", () => {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
    }, { once: true });
  }

  private renderKeys(): void {
    // 白鍵の描画
    WHITE_NOTES.forEach((note, index) => {
      const key = document.createElement("div");
      key.className = "white-key";
      key.dataset.note = note;

      const hint = document.createElement("div");
      hint.className = "key-hint";
      hint.textContent = Object.keys(KEY_MAP).find(k => KEY_MAP[k] === note)?.toUpperCase() || "";
      key.appendChild(hint);

      this.whiteKeysContainer.appendChild(key);
    });

    // 黒鍵の描画（位置計算）
    const whiteKeyCount = WHITE_NOTES.length;
    const whiteKeyWidth = 100 / whiteKeyCount;

    BLACK_NOTES.forEach(note => {
      const whiteIndex = WHITE_NOTES.findIndex(whiteNote => {
        const noteIndex = NOTES.indexOf(note);
        const whiteNoteIndex = NOTES.indexOf(whiteNote);
        return noteIndex === whiteNoteIndex + 1;
      });

      if (whiteIndex === -1) return;

      const key = document.createElement("div");
      key.className = "black-key";
      key.dataset.note = note;

      const left = (whiteIndex + 1) * whiteKeyWidth - (whiteKeyWidth * 0.4);
      key.style.left = `${left}%`;

      const hint = document.createElement("div");
      hint.className = "key-hint";
      hint.textContent = Object.keys(KEY_MAP).find(k => KEY_MAP[k] === note)?.toUpperCase() || "";
      hint.style.color = "#999";
      key.appendChild(hint);

      this.blackKeysContainer.appendChild(key);
    });
  }

  private setupEventListeners(): void {
    // マウスイベント
    this.piano.addEventListener("mousedown", this.handleStart.bind(this));
    this.piano.addEventListener("mouseup", this.handleEnd.bind(this));
    this.piano.addEventListener("mouseleave", this.handleEnd.bind(this));

    // タッチイベント
    this.piano.addEventListener("touchstart", this.handleTouchStart.bind(this));
    this.piano.addEventListener("touchend", this.handleTouchEnd.bind(this));
    this.piano.addEventListener("touchcancel", this.handleTouchEnd.bind(this));

    // キーボードイベント
    document.addEventListener("keydown", this.handleKeyDown.bind(this));
    document.addEventListener("keyup", this.handleKeyUp.bind(this));
  }

  private getNoteFromElement(el: HTMLElement): NoteName | null {
    return (el.dataset.note as NoteName) || null;
  }

  private playNote(note: NoteName): void {
    if (!this.audioContext) return;
    if (this.activeOscillators.has(note)) return; // 既に鳴っている場合は無視

    const osc = this.audioContext.createOscillator();
    const gain = this.audioContext.createGain();

    osc.type = "sine";
    osc.frequency.value = FREQUENCIES[note];
    gain.gain.value = 0.3;

    osc.connect(gain);
    gain.connect(this.audioContext.destination);

    osc.start();
    this.activeOscillators.set(note, osc);

    // 視覚的なフィードバック
    const key = this.piano.querySelector(`[data-note="${note}"]`) as HTMLElement;
    if (key) key.classList.add("active");
  }

  private stopNote(note: NoteName): void {
    const osc = this.activeOscillators.get(note);
    if (!osc || !this.audioContext) return;

    const gain = this.audioContext.createGain();
    osc.connect(gain);
    gain.connect(this.audioContext.destination);

    // 減衰させてから停止
    gain.gain.setValueAtTime(gain.gain.value, this.audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 0.1);
    osc.stop(this.audioContext.currentTime + 0.1);

    this.activeOscillators.delete(note);

    // 視覚的なフィードバック解除
    const key = this.piano.querySelector(`[data-note="${note}"]`) as HTMLElement;
    if (key) key.classList.remove("active");
  }

  // マウスイベントハンドラ
  private handleStart(e: MouseEvent): void {
    e.preventDefault();
    const target = e.target as HTMLElement;
    const note = this.getNoteFromElement(target);
    if (note) this.playNote(note);
  }

  private handleEnd(e: MouseEvent): void {
    e.preventDefault();
    const target = e.target as HTMLElement;
    const note = this.getNoteFromElement(target);
    if (note) this.stopNote(note);
  }

  // タッチイベントハンドラ
  private handleTouchStart(e: TouchEvent): void {
    e.preventDefault();
    const touch = e.touches[0];
    const target = document.elementFromPoint(touch.clientX, touch.clientY) as HTMLElement;
    const note = this.getNoteFromElement(target);
    if (note) this.playNote(note);
  }

  private handleTouchEnd(e: TouchEvent): void {
    e.preventDefault();
    // 簡易的にすべての音を止める
    this.activeOscillators.forEach((_, note) => this.stopNote(note));
  }

  // キーボードイベントハンドラ
  private handleKeyDown(e: KeyboardEvent): void {
    const key = e.key.toLowerCase();
    const note = KEY_MAP[key];
    if (note && !e.repeat) {
      e.preventDefault();
      this.playNote(note);
    }
  }

  private handleKeyUp(e: KeyboardEvent): void {
    const key = e.key.toLowerCase();
    const note = KEY_MAP[key];
    if (note) {
      e.preventDefault();
      this.stopNote(note);
    }
  }
}

// アプリ起動
document.addEventListener("DOMContentLoaded", () => {
  new PianoApp("piano");
});