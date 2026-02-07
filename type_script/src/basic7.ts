const canvas = document.getElementById('tetris') as HTMLCanvasElement;
const context = canvas.getContext('2d')!;
const scoreElement = document.getElementById('score')!;

context.scale(20, 20); // 1ブロックを20pxとして扱う

// ブロック（テトリミノ）の定義
const SHAPES = [
    [[0, 0, 0], [1, 1, 1], [0, 1, 0]], // T
    [[2, 2], [2, 2]],                // O
    [[0, 3, 3], [3, 3, 0], [0, 0, 0]], // S
    [[4, 4, 0], [0, 4, 4], [0, 0, 0]], // Z
    [[5, 0, 0], [5, 5, 5], [0, 0, 0]], // L
    [[0, 0, 6], [6, 6, 6], [0, 0, 0]], // J
    [[0, 0, 0, 0], [7, 7, 7, 7], [0, 0, 0, 0]] // I
];

const COLORS = [null, '#FF0D72', '#0DC2FF', '#0DFF72', '#F538FF', '#FF8E0D', '#FFE138', '#3877FF'];

// ゲームの状態
const arena = Array.from({length: 20}, () => Array(12).fill(0));
const player = {
    pos: {x: 5, y: 0},
    matrix: SHAPES[0],
    score: 0
};

// 描画関数
function draw() {
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    drawMatrix(arena, {x: 0, y: 0});
    drawMatrix(player.matrix, player.pos);
}

function drawMatrix(matrix: number[][], offset: {x: number, y: number}) {
    matrix.forEach((row, y) => {
        row.forEach((value, x) => {
            if (value !== 0) {
                context.fillStyle = COLORS[value]!;
                context.fillRect(x + offset.x, y + offset.y, 1, 1);
            }
        });
    });
}

// 衝突判定
function collide(arena: number[][], player: any) {
    const [m, o] = [player.matrix, player.pos];
    for (let y = 0; y < m.length; ++y) {
        for (let x = 0; x < m[y].length; ++x) {
            if (m[y][x] !== 0 && (arena[y + o.y] && arena[y + o.y][x + o.x]) !== 0) {
                return true;
            }
        }
    }
    return false;
}

// ライン消去
function arenaSweep() {
    outer: for (let y = arena.length - 1; y > 0; --y) {
        for (let x = 0; x < arena[y].length; ++x) {
            if (arena[y][x] === 0) continue outer;
        }
        const row = arena.splice(y, 1)[0].fill(0);
        arena.unshift(row);
        ++y;
        player.score += 10;
    }
    scoreElement.innerText = player.score.toString();
}

// 固定処理
function merge(arena: number[][], player: any) {
    player.matrix.forEach((row: number[], y: number) => {
        row.forEach((value, x) => {
            if (value !== 0) {
                arena[y + player.pos.y][x + player.pos.x] = value;
            }
        });
    });
}

// ブロックのリセット
function playerReset() {
    player.matrix = SHAPES[Math.floor(Math.random() * SHAPES.length)];
    player.pos.y = 0;
    player.pos.x = Math.floor(arena[0].length / 2) - Math.floor(player.matrix[0].length / 2);
    if (collide(arena, player)) {
        arena.forEach(row => row.fill(0));
        player.score = 0;
        scoreElement.innerText = "0";
    }
}

// 回転
function rotate(matrix: number[][]) {
    for (let y = 0; y < matrix.length; ++y) {
        for (let x = 0; x < y; ++x) {
            [matrix[x][y], matrix[y][x]] = [matrix[y][x], matrix[x][y]];
        }
    }
    matrix.forEach(row => row.reverse());
}

// 更新
let dropCounter = 0;
let lastTime = 0;
function update(time = 0) {
    const deltaTime = time - lastTime;
    lastTime = time;
    dropCounter += deltaTime;
    if (dropCounter > 1000) {
        playerDrop();
    }
    draw();
    requestAnimationFrame(update);
}

function playerDrop() {
    player.pos.y++;
    if (collide(arena, player)) {
        player.pos.y--;
        merge(arena, player);
        playerReset();
        arenaSweep();
    }
    dropCounter = 0;
}

// 入力イベント
document.addEventListener('keydown', event => {
    if (event.keyCode === 37) { // Left
        player.pos.x--;
        if (collide(arena, player)) player.pos.x++;
    } else if (event.keyCode === 39) { // Right
        player.pos.x++;
        if (collide(arena, player)) player.pos.x--;
    } else if (event.keyCode === 40) { // Down
        playerDrop();
    } else if (event.keyCode === 38) { // Up (Rotate)
        const pos = player.pos.x;
        let offset = 1;
        rotate(player.matrix);
        while (collide(arena, player)) {
            player.pos.x += offset;
            offset = -(offset + (offset > 0 ? 1 : -1));
            if (offset > player.matrix[0].length) {
                rotate(player.matrix); // 回転戻す
                player.pos.x = pos;
                return;
            }
        }
    }
});

playerReset();
update();