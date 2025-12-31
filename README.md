# AD-RecSys: Anomaly-Aware AI Recommendation System

**ユーザーの「いつもと違う」を検知し、最適な一歩を先回りして提案するレコメンデーションエンジン。**

## プロジェクト概要

本レポジトリは時系列や画像に異常検知を行い、いつもと違うことからユーザーにレコメンデーションを行うシステム構築を行います。

従来のレコメンドシステムは「過去の嗜好」に依存しますが、本システムはユーザーの行動パターンの「急激な変化（異常）」を検知します。これにより、ユーザーの潜在的なニーズの変化や、緊急性の高い需要を捉えたパーソナライズを実現します。

### 解決する課題

* **嗜好の固定化:** 過去のデータに縛られすぎたレコメンド（フィルターバブル）の解消。
* **状況変化への即応:** 生活環境や興味の急激な変化（例：旅行前、引越し、ライフイベント）を即座に検知。

## 基礎技術

### 異常検知
異常検知に関する詳細です。

- [anormaly_detect_techs/](anormaly_detect_techs/) : 異常検知調査した手法概要
- [anormaly_detect_techs/src](anormaly_detect_techs/src) : 異常検知の調査内容

<details>
<summary>
ここをクリックして詳細を表示（サブREADMEの内容）
</summary>
</details>

## 主な機能

* **Real-time Anomaly Detection**: LSTM/VAEを用いた行動パターンの異常検知。
* **Contextual Recommendation**: 検知された異常スコアに基づき、推薦アルゴリズムの重みを動的に調整。
* **Scalable Architecture**: 大規模な時系列ログを効率的に処理するデータパイプライン。
* **Visualization Dashboard**: 異常スコアとレコメンド精度の推移を可視化。

## デモ・視覚的イメージ

coming soon

## セットアップ手順

### インストール

coming soon

### クイックスタート

coming soon
