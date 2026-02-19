import Map from '@arcgis/core/Map';
import MapView from '@arcgis/core/views/MapView';
import FeatureLayer from '@arcgis/core/layers/FeatureLayer';
import config from '@arcgis/core/config';

// 1. APIキーの設定（ArcGIS Developer Dashboardで取得可能）
config.apiKey = "YOUR_ARCGIS_API_KEY";

class ArcGISManager {
    private view?: MapView;

    /**
     * マップの初期化
     * @param containerElement 埋め込み先のHTML要素ID
     */
    public async initMap(containerElement: string): Promise<void> {
        // マップのインスタンス作成（ベースマップの種類を指定）
        const map = new Map({
            basemap: "arcgis-topographic" // または "osm", "arcgis-navigation" など
        });

        // ビュー（表示画面）の設定
        this.view = new MapView({
            container: containerElement,
            map: map,
            center: [139.767, 35.681], // 東京付近 [経度, 緯度]
            zoom: 12
        });

        // 2. ArcGISデータ（FeatureLayer）との同期
        this.addSyncLayer(map);
    }

    /**
     * レイヤーの追加と同期
     */
    private addSyncLayer(map: Map): void {
        const featureLayer = new FeatureLayer({
            // 参照したいデータのURL（ArcGIS Online等のサービスURL）
            url: "https://services.arcgis.com/V6ZHFr6zdgNZuVG0/arcgis/rest/services/Landscape_Trees/FeatureServer/0",
            outFields: ["*"], // すべての属性を取得
            popupTemplate: {
                title: "{Name}",
                content: "種別: {Type}<br>状態: {Condition}"
            }
        });

        map.add(featureLayer);

        // データ読み込み完了時のイベントリスナー（同期確認用）
        featureLayer.when(() => {
            console.log("ArcGISデータとの同期が完了しました。");
        });
    }
}

// 実行
const arcgis = new ArcGISManager();
arcgis.initMap("viewDiv");