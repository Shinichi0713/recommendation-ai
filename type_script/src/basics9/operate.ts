import { readFile } from "fs/promises";

// テキストファイルを読み込んで行数を数える関数
async function countLines(filePath: string): Promise<number> {
  try {
    const content = await readFile(filePath, "utf-8");
    const lines = content.split("\n").filter(line => line.trim() !== "");
    return lines.length;
  } catch (error) {
    console.error("ファイルの読み込みに失敗しました:", error);
    return 0;
  }
}

// 使用例（例: sample.txt というファイルがある場合）
async function main() {
  const lineCount = await countLines("sample.txt");
  console.log(`ファイルの行数: ${lineCount}`);
}

main();

import { createServer, IncomingMessage, ServerResponse } from "http";

type RouteHandler = (req: IncomingMessage, res: ServerResponse) => void;

class SimpleServer {
  private routes: Map<string, RouteHandler> = new Map();

  get(path: string, handler: RouteHandler): void {
    this.routes.set(`GET ${path}`, handler);
  }

  post(path: string, handler: RouteHandler): void {
    this.routes.set(`POST ${path}`, handler);
  }

  private handleRequest(req: IncomingMessage, res: ServerResponse): void {
    const key = `${req.method} ${req.url}`;
    const handler = this.routes.get(key);

    if (handler) {
      handler(req, res);
    } else {
      res.statusCode = 404;
      res.end("Not Found");
    }
  }

  listen(port: number, callback?: () => void): void {
    const server = createServer((req, res) => this.handleRequest(req, res));
    server.listen(port, callback);
  }
}

// サーバーの設定と起動
const app = new SimpleServer();

app.get("/", (req, res) => {
  res.statusCode = 200;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.end("Hello, TypeScript Server!");
});

app.get("/time", (req, res) => {
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ currentTime: new Date().toISOString() }));
});

app.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});