import numpy as np
 
dataset = {
  'UserA': { 'Game1':2.0, 'Game2':4.0, 'Game3':1.0, 'Game4':5.0, 'Game5':1.0, 'Game6':0.0, 'Game7':0.0
  },
  'UserB': { 'Game1':2.0, 'Game2':3.0, 'Game3':1.0, 'Game4':0.0, 'Game5':0.0, 'Game6':4.0, 'Game7':5.0
  },
  'UserC': { 'Game1':5.0, 'Game2':1.0, 'Game3':5.0, 'Game4':0.0, 'Game5':0.0, 'Game6':0.0, 'Game7':2.0
  },
  'UserD': { 'Game1':4.0, 'Game2':1.0, 'Game3':4.0, 'Game4':3.0, 'Game5':5.0, 'Game6':0.0, 'Game7':0.0
  }
}

def getSimilarUsers(user1, user2):
    user1_data = list(dataset[user1].values())
    user2_data = list(dataset[user2].values())
 
    user1_eval_data = [user1_data[i] for i in range(0,3)]
    user2_eval_data = [user2_data[i] for i in range(0,3)]
    
    co = np.corrcoef(user1_eval_data, user2_eval_data)[0,1]
 
    return(abs(co))
 
def getRecommendGame(user):
    similar_user = ''
    best_sim_score = 0.0
    best_sim_g_score = 0.0
    # 自分以外のユーザーデータを取得
    not_user_dataset = dataset.copy()
    user_dataset = not_user_dataset.pop(user)
    user_not_play_games = [k for k, v in user_dataset.items() if v == 0.0]
    
    # 自分と一番近いユーザーを探索
    for u in not_user_dataset.keys():
        tmp_score = getSimilarUsers(user, u)
        
        if best_sim_score <= tmp_score:
            similar_user = u
            best_sim_score = tmp_score
    # 一番近いユーザーの中で、自分がプレイしていないゲームの中で一番評価が高いゲームを推薦
    for g in user_not_play_games:
        tmp_g_score = not_user_dataset[similar_user][g]
        
        if best_sim_g_score <= tmp_g_score:
            recommend_game = g
            best_sim_g_score = tmp_g_score
    
    return(recommend_game)
   
print(getRecommendGame('UserA'))