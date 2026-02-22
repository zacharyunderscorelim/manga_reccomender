import torch
import torch.nn as nn
import pandas as pd
from model import MatrixFactorisation

checkpoint = torch.load('model.pth', weights_only=False)
manga_to_idx = checkpoint['manga_to_idx']
user_to_idx = checkpoint['user_to_idx']
n_manga = checkpoint['n_manga']
n_users = checkpoint['n_users']

idx_to_manga = {idx: manga for manga, idx in manga_to_idx.items()}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}

model = MatrixFactorisation(n_manga, n_users)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

manga_df = pd.read_csv('manga_dataset.csv')

reccos = []
def get_reccos(username, model, top_k=10):
    try:
        user_idx = user_to_idx[username]
    except Exception as e:
        print("Username not found")
        return
    manga_idxs, predicted_scores = model.predict_for_user(user_idx, n_manga, top_k)
    for idx, score in zip(manga_idxs, predicted_scores):
        manga_id = idx_to_manga[idx]
        manga_info = manga_df[manga_df['manga_id'] == manga_id].iloc[0]
        reccos.append({
            'manga_id': manga_id,
            'title': manga_info.get('title', 'Unknown'),
            'predicted_score': score
        })
        if len(reccos) >= top_k:
            break
    return pd.DataFrame(reccos)

if __name__ == '__main__':
    username = "chairmanping"
    recs_df = get_reccos(username, model, top_k=50)
    print(f"\nRecommendations for {username}:")
    print(recs_df)