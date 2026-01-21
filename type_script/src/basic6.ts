import ping from 'ping';
import * as fs from 'fs';

// 1. 監視対象のリスト（IPアドレスやドメイン名）
const hosts = ['8.8.8.8', 'google.com', '192.168.1.1', 'invalid.host'];

// 2. ログファイルのパス
const logFile = './network_status.log';

async function checkNetwork() {
    console.log('--- 監視開始 ---');
    const timestamp = new Date().toLocaleString();
    let report = `--- 記録日時: ${timestamp} ---\n`;

    // 全てのホストに対して順番にPingを実行
    for (const host of hosts) {
        try {
            // Pingを実行（タイムアウト3秒設定）
            const res = await ping.promise.probe(host, { timeout: 3 });
            
            const status = res.alive ? '[OK]' : '[NG]';
            const responseTime = res.time !== 'unknown' ? `${res.time}ms` : '---';
            
            const line = `${status} ${host.padEnd(15)} | 応答時間: ${responseTime}`;
            console.log(line);
            report += line + '\n';
        } catch (error) {
            report += `[ERR] ${host}: エラー発生\n`;
        }
    }

    report += '\n';

    // 3. 結果をログファイルに追記（自動保存）
    fs.appendFileSync(logFile, report, 'utf8');
    console.log('--- 監視完了。ログに保存しました ---');
}

// 実行
checkNetwork();