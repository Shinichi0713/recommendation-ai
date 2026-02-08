// 定数
const GRAVITY = 0.5; // 重力加速度
const JUMP_POWER = -10; // ジャンプ力 (Y軸は下がプラスなのでマイナス)
const MOVE_SPEED = 5; // 移動速度
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;

// Canvasの設定
const canvas = document.getElementById('gameCanvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;

// プレイヤーの状態
interface Player {
    x: number;
    y: number;
    width: number;
    height: number;
    velX: number; // X方向の速度
    velY: number; // Y方向の速度
    onGround: boolean; // 接地しているか
}

const player: Player = {
    x: 50,
    y: CANVAS_HEIGHT - 50, // 最初は地面に立っている
    width: 30,
    height: 30,
    velX: 0,
    velY: 0,
    onGround: false,
};

// プラットフォーム（地面やブロック）
interface Platform {
    x: number;
    y: number;
    width: number;
    height: number;
}

const platforms: Platform[] = [
    // 地面
    { x: 0, y: CANVAS_HEIGHT - 30, width: CANVAS_WIDTH, height: 30 },
    // 浮いているブロック1
    { x: 150, y: CANVAS_HEIGHT - 150, width: 100, height: 20 },
    // 浮いているブロック2
    { x: 400, y: CANVAS_HEIGHT - 250, width: 150, height: 20 },
    // 壁のようなブロック
    { x: 600, y: CANVAS_HEIGHT - 100, width: 50, height: 70 },
];

// キー入力の状態
const keys: { [key: string]: boolean } = {};
document.addEventListener('keydown', (e) => { keys[e.code] = true; });
document.addEventListener('keyup', (e) => { keys[e.code] = false; });

// ゲームループ
function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

// 更新処理 (物理演算、移動など)
function update() {
    // プレイヤーの左右移動
    player.velX = 0;
    if (keys['ArrowLeft']) {
        player.velX = -MOVE_SPEED;
    }
    if (keys['ArrowRight']) {
        player.velX = MOVE_SPEED;
    }

    // ジャンプ
    if (keys['Space'] && player.onGround) {
        player.velY = JUMP_POWER;
        player.onGround = false;
    }

    // 重力適用
    player.velY += GRAVITY;

    // プレイヤーの位置更新
    player.x += player.velX;
    player.y += player.velY;

    // プレイヤーが画面外に出ないように制限
    if (player.x < 0) player.x = 0;
    if (player.x + player.width > CANVAS_WIDTH) player.x = CANVAS_WIDTH - player.width;

    // プラットフォームとの衝突判定
    player.onGround = false; // 接地状態を一旦リセット
    platforms.forEach(platform => {
        // AABB (Axis-Aligned Bounding Box) 衝突判定
        if (
            player.x < platform.x + platform.width &&
            player.x + player.width > platform.x &&
            player.y < platform.y + platform.height &&
            player.y + player.height > platform.y
        ) {
            // 衝突している場合
            // 下方向への衝突（プラットフォームの上にいる）
            if (player.velY > 0 && player.y + player.height > platform.y && player.y < platform.y) {
                player.y = platform.y - player.height; // プラットフォームのY座標に合わせる
                player.velY = 0; // Y方向の速度をリセット
                player.onGround = true; // 接地状態にする
            }
            // 上方向への衝突（ブロックの下からぶつかる）
            else if (player.velY < 0 && player.y < platform.y + platform.height && player.y + player.height > platform.y + platform.height) {
                player.y = platform.y + platform.height;
                player.velY = 0;
            }
            // 左右方向への衝突 (今回は簡易的に停止させるだけ)
            // この部分はマリオだと壁に張り付くような挙動になるが、今回はシンプルに
            else if (player.velX > 0 && player.x + player.width > platform.x && player.x < platform.x) {
                player.x = platform.x - player.width;
                player.velX = 0;
            } else if (player.velX < 0 && player.x < platform.x + platform.width && player.x + player.width > platform.x + platform.width) {
                player.x = platform.x + platform.width;
                player.velX = 0;
            }
        }
    });

    // 画面下部に落ちた場合の処理
    if (player.y + player.height > CANVAS_HEIGHT - 30 && !player.onGround) {
        // 地面に衝突していなかったが画面下部に到達した場合 (本来の地面より下に行った場合)
        player.y = CANVAS_HEIGHT - 30 - player.height; // 地面に設置
        player.velY = 0;
        player.onGround = true;
    }
}

// 描画処理
function draw() {
    // 画面クリア (空色にする)
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.fillStyle = '#87CEEB';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // プラットフォーム描画
    ctx.fillStyle = '#8B4513'; // 茶色
    platforms.forEach(platform => {
        ctx.fillRect(platform.x, platform.y, platform.width, platform.height);
    });

    // プレイヤー描画 (赤色)
    ctx.fillStyle = '#FF0000';
    ctx.fillRect(player.x, player.y, player.width, player.height);
}

// ゲーム開始
gameLoop();