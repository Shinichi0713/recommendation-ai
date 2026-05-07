// src/cli.ts
import { countLines } from './fileLineCounter';

function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('使用方法: npx ts-node src/cli.ts <ファイル名>');
    process.exit(1);
  }

  const filePath = args[0];

  try {
    const lineCount = countLines(filePath);
    console.log(`ファイル "${filePath}" の行数: ${lineCount}`);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}

main();