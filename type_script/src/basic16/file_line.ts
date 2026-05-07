// src/fileLineCounter.ts
import { readFileSync } from 'fs';

export function countLines(filePath: string): number {
  try {
    const content = readFileSync(filePath, 'utf-8');
    // 空行も1行としてカウント
    const lines = content.split('\n');
    return lines.length;
  } catch (error) {
    throw new Error(`ファイルの読み込みに失敗しました: ${error.message}`);
  }
}