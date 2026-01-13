// 型定義: Windowオブジェクトに非標準のAPIを拡張（必要に応じて）
interface Window {
  showDirectoryPicker: () => Promise<FileSystemDirectoryHandle>;
}

const pickerBtn = document.getElementById('picker') as HTMLButtonElement;
const gallery = document.getElementById('gallery') as HTMLDivElement;

pickerBtn.addEventListener('click', async () => {
  try {
    // 1. ユーザーにフォルダ選択ダイアログを表示
    const dirHandle = await window.showDirectoryPicker();
    
    // ギャラリーをクリア
    gallery.innerHTML = '';

    // 2. フォルダ内のファイルを反復処理
    for await (const entry of dirHandle.values()) {
      if (entry.kind === 'file') {
        const file = await entry.getFile();
        
        // 3. 画像ファイルかどうかをチェック (MIMEタイプ)
        if (file.type.startsWith('image/')) {
          displayImage(file);
        }
      }
    }
  } catch (err) {
    console.error('フォルダの読み込みに失敗しました:', err);
  }
});

/**
 * ファイルを読み込んで画面に表示するヘルパー関数
 */
function displayImage(file: File): void {
  const reader = new FileReader();

  reader.onload = (e) => {
    const container = document.createElement('div');
    const img = document.createElement('img');
    const label = document.createElement('p');

    img.src = e.target?.result as string;
    img.style.width = '200px';
    img.style.height = '150px';
    img.style.objectFit = 'cover';
    img.style.borderRadius = '8px';

    label.textContent = file.name;
    label.style.fontSize = '12px';
    label.style.textAlign = 'center';

    container.appendChild(img);
    container.appendChild(label);
    gallery.appendChild(container);
  };

  reader.readAsDataURL(file);
}