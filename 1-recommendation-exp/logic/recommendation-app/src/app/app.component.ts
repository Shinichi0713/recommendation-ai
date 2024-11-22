// src/app/app.component.ts
import { Component } from '@angular/core';
import axios from 'axios';
import { NgFor } from '@angular/common';
import { FormsModule } from '@angular/forms';  // FormsModule をインポート

// Recommendationインターフェースを定義(ないとエラーになる)
interface Recommendation {
  item: string;
  score: number;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  standalone: true, // スタンドアロンコンポーネントとしてマーク
  imports: [NgFor, FormsModule] // NgForをインポート
})
export class AppComponent {
  title = 'recommendation-app';
  items = ["Game1", "Game2", "Game3", "Game4"];
  ratings: {[key: string]: number}={};
  recommendations: Recommendation[] = [];
  recommendation: string = '';

  constructor() {
    this.items.forEach(item => {
      this.ratings[item] = 0;
    });
  }

  async submitRatings() {
    const user = 'User4'; // 固定ユーザー名（本来は動的に取得する）
    const dataRequest = {
      user,
      ratings: this.ratings
    };
    console.log(dataRequest);
    await axios.post('http://localhost:3000/rate', dataRequest);
    const response = await axios.get(`http://localhost:3000/recommend/${user}`);
    console.log(response);
    this.recommendation = response.data.data;
    console.log(this.recommendations);
  }
}
