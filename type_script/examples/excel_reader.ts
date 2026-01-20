import * as XLSX from 'xlsx';
import * as fs from 'fs';

// 設定：読み込むエクセルファイルと出力するCSVファイル名
const inputFilePath = './sample.xlsx';  // 読み込みたいファイル名に変更してください
const outputCsvPath = './sheet_list.csv';

function exportSheetNamesToCsv(inputPath: string, outputPath: string): void {
    try {
        // 1. ローカルのエクセルファイルを読み込む
        // cellFormulaなどは不要なので、メモリ節約のため最小限の設定で読み込み
        const workbook = XLSX.readFile(inputPath, { bookSheets: true });

        // 2. シート名のリストを取得
        const sheetNames = workbook.SheetNames;

        if (sheetNames.length === 0) {
            console.log("シートが見つかりませんでした。");
            return;
        }

        // 3. CSV形式の文字列を作成
        // ヘッダーを付けて、各シート名を改行で繋ぐ
        const csvContent = "Sheet Index,Sheet Name\n" + 
            sheetNames.map((name, index) => `${index + 1},${name}`).join('\n');

        // 4. CSVファイルとして出力
        fs.writeFileSync(outputPath, csvContent, 'utf8');

        console.log(`成功: シートリストを "${outputPath}" に出力しました。`);
        console.log("シート一覧:", sheetNames);

    } catch (error) {
        console.error("エラーが発生しました:", error);
    }
}

// 実行
exportSheetNamesToCsv(inputFilePath, outputCsvPath);