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
  items = ["Item1", "Item2", "Item3", "Item4"];
  ratings: {[key: string]: number}={};
  recommendations: Recommendation[] = [];

  constructor() {
    this.items.forEach(item => {
      this.ratings[item] = 0;
    });
  }

  async submitRatings() {
    const user = 'User1'; // 固定ユーザー名（本来は動的に取得する）
    await axios.post('http://localhost:3000/rate', {
      user,
      ratings: this.ratings
    });
    const response = await axios.get(`http://localhost:3000/recommend/${user}`);
    console.log(response);
    this.recommendations = response.data;
  }
}
