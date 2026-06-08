<!-- 再生ボタン -->
<button id="bgm-play-btn">BGMを再生</button>

<!-- YouTubeプレーヤーを表示するコンテナ（非表示にしてもOK） -->
<div id="youtube-player-container"></div>

<script>
// YouTubeプレーヤーオブジェクト
var player;

// APIスクリプトを読み込む
var tag = document.createElement('script');
tag.src = "https://www.youtube.com/iframe_api";
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

// API読み込み完了時に呼ばれる
function onYouTubeIframeAPIReady() {
  player = new YT.Player('youtube-player-container', {
    height: '0',   // 高さ0で非表示にする（BGM専用）
    width: '0',    // 幅も0で非表示
    videoId: '3_yuaHEVn4k', // ここにBGMにしたい動画IDを入れる
    playerVars: {
      'autoplay': 0,      // 自動再生はしない（ボタンで開始）
      'mute': 0,          // 最初から音声あり
      'playsinline': 1,   // モバイルでインライン再生
      'enablejsapi': 1,   // APIを有効化
      'controls': 0,       // コントロール非表示
      'rel': 0            // 関連動画非表示
    },
    events: {
      'onReady': onPlayerReady,
      'onStateChange': onPlayerStateChange
    }
  });
}

// プレーヤー準備完了
function onPlayerReady(event) {
  console.log('プレーヤー準備完了');
  // ここでは自動再生はしない
}

// プレーヤーの状態変化
function onPlayerStateChange(event) {
  // 必要に応じて再生・停止などの処理を追加
}

// ボタンクリックでBGM再生
document.getElementById('bgm-play-btn').addEventListener('click', function() {
  if (player && typeof player.playVideo === 'function') {
    player.playVideo();
    player.setVolume(50); // 音量を50%に設定（0〜100）
  }
});
</script>


<button id="bgm-trigger" style="padding: 10px 20px; font-size: 16px; background-color: #008cba; color: white; border: none; border-radius: 5px; cursor: pointer;">
  🎵 BGMを再生する
</button>

<div style="width: 1px; height: 1px; overflow: hidden; opacity: 0; position: absolute;">
  <iframe id="bgm-player" width="100" height="100" 
  src="https://www.youtube.com/embed/3_yuaHEVn4k?enablejsapi=1" 
  title="BGM Player" frameborder="0" allow="autoplay">
  </iframe>
</div>

<script>
  document.getElementById('bgm-trigger').addEventListener('click', function() {
    var iframe = document.getElementById('bgm-player');
    
    // すでに再生中の場合は一時停止、停止中の場合は再生するトグル処理
    if (this.innerText.includes('再生')) {
      // JavaScript API経由で再生コマンドを送信
      iframe.contentWindow.postMessage('{"event":"command","func":"playVideo","args":""}', '*');
      this.innerText = '⏸ BGMを停止する';
      this.style.backgroundColor = '#f44336'; // ボタンの色を赤系に
    } else {
      iframe.contentWindow.postMessage('{"event":"command","func":"pauseVideo","args":""}', '*');
      this.innerText = '🎵 BGMを再生する';
      this.style.backgroundColor = '#008cba'; // ボタンの色を青系に
    }
  });
</script>


<iframe width="1537" height="864" src="https://www.youtube.com/embed/SFCCZ89dJQU?list=RDSFCCZ89dJQU" title="喫茶店ジャズ音楽｜ゆっくりしたい日におすすめ、静かで落ち着く癒しのプレイリスト［カフェ・リラックス・作業用bgm］- Relaxing Smooth Jazz -" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<iframe width="1537" height="864" src="https://www.youtube.com/embed/3_yuaHEVn4k?list=RD3_yuaHEVn4k" title="紅の豚より　マルコとジーナのテーマ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>