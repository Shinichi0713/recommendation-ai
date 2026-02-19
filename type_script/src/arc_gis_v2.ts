import React, { useEffect, useRef, useState } from 'react';
import Map from '@arcgis/core/Map';
import MapView from '@arcgis/core/views/MapView';
import FeatureLayer from '@arcgis/core/layers/FeatureLayer';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import Graphic from '@arcgis/core/Graphic';
import Point from '@arcgis/core/geometry/Point';
import SimpleMarkerSymbol from '@arcgis/core/symbols/SimpleMarkerSymbol';
import SimpleRenderer from '@arcgis/core/renderers/SimpleRenderer';
import config from '@arcgis/core/config';

// 🚨🚨🚨 YOUR_ARCGIS_API_KEY をあなたのAPIキーに置き換えてください 🚨🚨🚨
config.apiKey = "YOUR_ARCGIS_API_KEY";

// 🚨🚨🚨 YOUR_FEATURE_LAYER_URL をあなたの編集可能なFeatureLayerのURLに置き換えてください 🚨🚨🚨
// 例: ArcGIS Onlineで新規にFeature Layerを作成し、編集権限を付与してください
const DANGER_REPORT_LAYER_URL = "YOUR_FEATURE_LAYER_URL_FOR_DANGER_REPORTS";

const SafetyMapApp: React.FC = () => {
    const mapDiv = useRef<HTMLDivElement>(null);
    const [view, setView] = useState<MapView | null>(null);
    const [isReporting, setIsReporting] = useState<boolean>(false);
    const [reportText, setReportText] = useState<string>('');
    const [lastClickLocation, setLastClickLocation] = useState<Point | null>(null);

    useEffect(() => {
        if (mapDiv.current) {
            const map = new Map({
                basemap: "arcgis-topographic"
            });

            const graphicsLayer = new GraphicsLayer(); // クリック地点の一時表示用
            map.add(graphicsLayer);

            // 報告用FeatureLayer (既存のURLを指す)
            const dangerReportLayer = new FeatureLayer({
                url: DANGER_REPORT_LAYER_URL,
                outFields: ["*"],
                // AI分析後の危険度に応じてシンボルを動的に変更
                renderer: new SimpleRenderer({
                    symbol: new SimpleMarkerSymbol({
                        size: 8,
                        color: [255, 0, 0, 0.7], // デフォルトは赤
                        outline: { width: 1, color: [255, 255, 255, 0.8] }
                    })
                }),
                popupTemplate: {
                    title: "危険報告",
                    content: "報告内容: {report_text}<br>危険度: {danger_level}" // AIが分類したdanger_levelを表示
                },
                // データをリアルタイムで更新（例えば1分おき）
                refreshInterval: 1 
            });
            map.add(dangerReportLayer);

            const mapView = new MapView({
                container: mapDiv.current,
                map: map,
                center: [139.767, 35.681],
                zoom: 12
            });

            mapView.when(() => {
                console.log("マップがロードされました。");
                setView(mapView);
            });

            // マップクリックイベントで報告場所を決定
            mapView.on("click", (event) => {
                if (isReporting) {
                    graphicsLayer.removeAll(); // 前のクリック点をクリア
                    const point = event.mapPoint;
                    setLastClickLocation(point);
                    const clickedGraphic = new Graphic({
                        geometry: point,
                        symbol: new SimpleMarkerSymbol({
                            size: 10,
                            color: [0, 191, 255, 0.8], // クリック点は青
                            outline: { width: 1, color: [255, 255, 255, 0.8] }
                        })
                    });
                    graphicsLayer.add(clickedGraphic);
                }
            });

            return () => mapView && mapView.destroy();
        }
    }, [isReporting]);

    // 危険報告を送信する関数
    const submitReport = async () => {
        if (!lastClickLocation || !reportText) {
            alert('場所をクリックし、報告内容を入力してください。');
            return;
        }

        // ここで、サーバーサイドのAI処理にレポートを送信
        // 例: fetch('/api/analyze-report', { method: 'POST', body: JSON.stringify({ text: reportText }) })
        // AIが 'danger_level' を返すとする

        const dangerReportLayer = view?.map?.allLayers.find(layer => 
            layer.type === 'feature' && (layer as FeatureLayer).url === DANGER_REPORT_LAYER_URL
        ) as FeatureLayer;

        if (dangerReportLayer) {
            const newGraphic = new Graphic({
                geometry: lastClickLocation,
                attributes: {
                    report_text: reportText,
                    danger_level: '中', // 🚨🚨🚨 ここはAI解析結果で置き換える 🚨🚨🚨
                    timestamp: new Date().toISOString()
                }
            });

            try {
                // FeatureLayerに新しいフィーチャを追加
                await dangerReportLayer.applyEdits({
                    addFeatures: [newGraphic]
                });
                alert('危険情報を送信しました！');
                setIsReporting(false);
                setReportText('');
                setLastClickLocation(null);
                view?.map?.findLayerById('graphicsLayer')?.removeAll(); // 一時表示グラフィックをクリア
            } catch (error) {
                console.error("レポート送信エラー:", error);
                alert('レポートの送信に失敗しました。');
            }
        }
    };

    return (
        <div style={{ height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '10px', backgroundColor: '#f0f0f0', borderBottom: '1px solid #ccc' }}>
                <h2>地域安全マップ ({isReporting ? '報告モード' : '閲覧モード'})</h2>
                <button onClick={() => setIsReporting(!isReporting)}>
                    {isReporting ? '報告モードを終了' : '危険を報告する'}
                </button>
                {isReporting && (
                    <div style={{ marginTop: '10px', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
                        <p>マップ上の危険な場所をクリックし、内容を入力してください。</p>
                        <textarea
                            placeholder="何があったか具体的に入力してください..."
                            value={reportText}
                            onChange={(e) => setReportText(e.target.value)}
                            style={{ width: '90%', minHeight: '60px', marginTop: '5px' }}
                        />
                        <button onClick={submitReport} style={{ marginLeft: '10px' }}>
                            報告を送信
                        </button>
                    </div>
                )}
            </div>
            <div id="viewDiv" ref={mapDiv} style={{ flexGrow: 1 }} />
        </div>
    );
};

export default SafetyMapApp;