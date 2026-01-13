// Node.js (fsモジュール) を使う場合の例
import fs from 'fs';
import path from 'path';

const folderPath = './images';
const files = fs.readdirSync(folderPath);

const imageFiles = files.filter(file => 
  ['.jpg', '.png', '.webp'].includes(path.extname(file).toLowerCase())
);
// この後、パス一覧をJSONでフロントに返す処理が必要です