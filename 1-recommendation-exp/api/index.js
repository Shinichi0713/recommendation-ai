// server.js
const express = require('express');
const bodyParser = require('body-parser');
const app = express();

app.use(bodyParser.json());

// サンプルデータ
const data = {
  items: ["Item1", "Item2", "Item3", "Item4"],
  users: {
    "User1": {"Item1": 5, "Item2": 3, "Item3": 4},
    "User2": {"Item1": 4, "Item2": 5, "Item3": 2, "Item4": 5},
    "User3": {"Item1": 2, "Item2": 1, "Item3": 5, "Item4": 3}
  }
};

// ユーザーの評価を受け取るエンドポイント
app.post('/rate', (req, res) => {
  const { user, ratings } = req.body;
  data.users[user] = ratings;
  res.send('Ratings received');
});

// 協調フィルタリングによるおすすめを計算するエンドポイント
app.get('/recommend/:user', (req, res) => {
  const user = req.params.user;
  const recommendations = getRecommendations(user);
  res.json(recommendations);
});

// 協調フィルタリングのアルゴリズム（簡易版）
function getRecommendations(user) {
  const userRatings = data.users[user];
  const scores = {};
  const totals = {};

  for (let otherUser in data.users) {
    if (otherUser === user) continue;
    const otherRatings = data.users[otherUser];

    for (let item in otherRatings) {
      if (!(item in userRatings)) {
        if (!(item in scores)) {
          scores[item] = 0;
          totals[item] = 0;
        }
        scores[item] += otherRatings[item];
        totals[item] += 1;
      }
    }
  }

  const recommendations = [];
  for (let item in scores) {
    recommendations.push({ item, score: scores[item] / totals[item] });
  }

  recommendations.sort((a, b) => b.score - a.score);
  return recommendations;
}

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});