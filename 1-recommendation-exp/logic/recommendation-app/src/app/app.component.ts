// src/app/app.component.ts
import { Component } from '@angular/core';
import axios from 'axios';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  standalone: true, // スタンドアロンコンポーネントとしてマーク
})
export class AppComponent {
  title = 'recommendation-app';
  items = ["Item1", "Item2", "Item3", "Item4"];
  ratings: {[key: string]: number}={};
  recommendations = [];

  constructor() {
    this.items.forEach(item => {
      this.ratings[item] = 0;
    });
  }

  async submitRatings() {
    const user = 'User4'; // 固定ユーザー名（本来は動的に取得する）
    await axios.post('http://localhost:3000/rate', {
      user,
      ratings: this.ratings
    });
    const response = await axios.get(`http://localhost:3000/recommend/${user}`);
    this.recommendations = response.data;
  }
}
