import Sitemapper from 'sitemapper';

/**
 * 指定されたドメインの全パス（URL）を取得する関数
 * @param domain 取得対象のドメイン (例: https://example.com)
 */
async function getAllPaths(domain: string): Promise<string[]> {
    // サイトマップの場所を推定（標準的な場所）
    const sitemapUrl = `${domain}/sitemap.xml`;
    
    const sitemapper = new Sitemapper({
        url: sitemapUrl,
        timeout: 15000, // 15秒でタイムアウト
    });

    try {
        console.log(`Fetching sitemap from: ${sitemapUrl}...`);
        const { sites } = await sitemapper.fetch();

        if (!sites || sites.length === 0) {
            console.warn('サイトマップが見つからないか、URLが空です。');
            return [];
        }

        // フルURLからパス部分のみを抽出したい場合は以下のように加工
        const paths = sites.map(url => {
            const urlObj = new URL(url);
            return urlObj.pathname;
        });

        return paths;
    } catch (error) {
        console.error('エラーが発生しました:', error);
        return [];
    }
}

// 実行例
const targetDomain = 'https://ai-parenting-blog.example.com'; // ご自身のサイトURLなどに書き換えてください
getAllPaths(targetDomain).then(paths => {
    console.log(`取得したパス一覧 (${paths.length}件):`);
    console.log(paths);
});


import { exec } from 'child_process';
exec('powershell.exe -Command "Get-Process"', (error, stdout) => {
    console.log(stdout);
});