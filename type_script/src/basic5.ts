const upload = document.getElementById('upload') as HTMLInputElement;
const originalCanvas = document.getElementById('originalCanvas') as HTMLCanvasElement;
const edgeCanvas = document.getElementById('edgeCanvas') as HTMLCanvasElement;
const ctxOrig = originalCanvas.getContext('2d', { willReadFrequently: true })!;
const ctxEdge = edgeCanvas.getContext('2d')!;

upload.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            // キャンバスサイズを画像に合わせる
            originalCanvas.width = edgeCanvas.width = img.width;
            originalCanvas.height = edgeCanvas.height = img.height;

            // 元画像を描画
            ctxOrig.drawImage(img, 0, 0);

            // エッジ検知処理の実行
            applyEdgeDetection();
        };
        img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
});

function applyEdgeDetection() {
    const width = originalCanvas.width;
    const height = originalCanvas.height;
    const imageData = ctxOrig.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    // 出力用のバッファ
    const output = ctxEdge.createImageData(width, height);
    const outData = output.data;

    // ラプラシアンフィルタのカーネル
    const kernel = [
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    ];

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let r = 0, g = 0, b = 0;

            // 3x3の畳み込み演算
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const pos = ((y + ky) * width + (x + kx)) * 4;
                    const weight = kernel[(ky + 1) * 3 + (kx + 1)];
                    r += data[pos] * weight;
                    g += data[pos + 1] * weight;
                    b += data[pos + 2] * weight;
                }
            }

            const idx = (y * width + x) * 4;
            // 計算結果をグレースケールとして出力
            const gray = (r + g + b) / 3;
            outData[idx] = outData[idx + 1] = outData[idx + 2] = gray;
            outData[idx + 3] = 255; // 不透明度
        }
    }
    ctxEdge.putImageData(output, 0, 0);
}