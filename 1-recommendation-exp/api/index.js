const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();

app.use(bodyParser.json());
app.use(cors());

// サンプルデータ
const data = {
  items: ["Game1", "Game2", "Game3", "Game4"],
  users: {
    "User1": {"Game1": 5, "Game2": 3, "Game3": 4, "Game4": 0},
    "User2": {"Game1": 4, "Game2": 5, "Game3": 2, "Game4": 5},
    "User3": {"Game1": 2, "Game2": 1, "Game3": 5, "Game4": 3}
  }
};

let userRatings = {};

// ユーザーの評価を受け取るエンドポイント
app.post('/rate', (req, res) => {
  // console.log(req);
  const { user, ratings } = req.body;
  userRatings = ratings;
  console.log(userRatings);
  res.send('Ratings received');
});

// 協調フィルタリングによるおすすめを計算するエンドポイント
app.get('/recommend/:user', (req, res) => {
  const user = req.params.user;
  // if (!data.users[user]) {
  //   return res.status(404).json({ error: 'User not found' });
  // }
  console.log(userRatings);
  const recommendations = getRecommendations(user);
  console.log(recommendations);
  res.json(recommendations);
});

// 辞書の値を配列化する関数
function getDictionaryValues(obj) {
  return Object.keys(obj).map(key => obj[key]);
}

/**
 * 2つのベクトルのコサイン類似度を計算する関数
 * @param vecA - ベクトルA
 * @param vecB - ベクトルB
 * @returns コサイン類似度（-1 から 1 の範囲）
 */
function cosineSimilarity(vecA, vecB){
  if (vecA.length !== vecB.length) {
    throw new Error('ベクトルの長さが一致していません');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  if (normA === 0 || normB === 0) {
    throw new Error('ゼロベクトルが含まれています');
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}


// 協調フィルタリングのアルゴリズム（簡易版）
function getRecommendations(userData) {
  let similarity_best = -1.0;
  let user_best = "";
  console.log(userRatings);
  for (let otherUser in data.users) {
    if (otherUser === userData.user) continue;
    const otherRatings = data.users[otherUser];
    console.log(otherRatings);
    similarity = cosineSimilarity(getDictionaryValues(userRatings), getDictionaryValues(otherRatings));
    if (similarity > similarity_best) {
      similarity_best = similarity;
      user_best = otherUser;
    }
  }

  // 一番、自分と類似ユーザーとの乖離が激しいアイテムを推薦
  const otherRatings = data.users[user_best];
  let most_different = -100;
  let recommendation = "";
  for (let item in otherRatings) {
    console.log(otherRatings[item]-userRatings[item]);
    if (otherRatings[item]-userRatings[item] > most_different) {
      most_different = otherRatings[item]-userRatings[item];
      recommendation = item;
    }
  }

  return {data: recommendation, similarity: similarity_best};
}

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});



// 推薦結果の表示
// similarity = cosineSimilarity(getDictionaryValues(data.users["User1"]), getDictionaryValues(data.users["User2"]));
// console.log(similarity);
// const dataUser = {
//   user: "Allice",
//   ratings: {"Game1": 5, "Game2": 3, "Game3": 4, "Game4": 0}
// };
// const recommendations = getRecommendations(dataUser);
// console.log(`Recommendations :`, recommendations);