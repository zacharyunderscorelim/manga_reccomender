import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split

class MatrixFactorisation(nn.Module):
    def __init__(self, n_manga, n_users):
        super().__init__()
        self.embedding_dim = 64 #number of latent features

        self.manga_embeddings = nn.Embedding(n_manga, self.embedding_dim, dtype=torch.float64)
        self.user_embeddings = nn.Embedding(n_users, self.embedding_dim, dtype=torch.float64)
        self.dropout = nn.Dropout(0.2)
        self.user_b = nn.Embedding(n_users, 1, dtype=torch.float64)
        self.manga_b = nn.Embedding(n_manga, 1, dtype=torch.float64)
        self.global_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        nn.init.normal_(self.manga_embeddings.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.user_b.weight)
        nn.init.zeros_(self.manga_b.weight)

        self.loss_fn = nn.MSELoss()
    
    def forward(self, xs):
        x_manga = xs[:, 0]
        x_user = xs[:, 1]
        y_preds = (self.dropout(self.manga_embeddings(x_manga)) * self.dropout(self.user_embeddings(x_user))).sum(dim=1)
        y_preds = y_preds + self.user_b(x_user).squeeze()
        y_preds = y_preds + self.manga_b(x_manga).squeeze()
        y_preds = y_preds + self.global_b
        return y_preds
    
    def train_loop(self, xs, ys, xs_val, ys_val, epochs, lr, decay):
        optimiser = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=decay)
        val_losses = []
        overfit_count = 0
        for i in range(epochs):
            self.train()
            optimiser.zero_grad()

            y_preds = self(xs)
            loss = self.loss_fn(y_preds, ys)

            loss.backward()
            optimiser.step()

            self.eval()
            with torch.no_grad():
                val_preds = self(xs_val)
                val_loss = self.loss_fn(val_preds, ys_val)
                val_losses.append(val_loss)
            
            if i%10 == 0:
                print(f"Epoch {i}, loss = {loss}, val loss = {val_loss}")
                if len(val_losses) > 50 and val_losses[-1] > val_losses[-50]:
                    overfit_count += 1
                else:
                    overfit_count = 0

            if overfit_count >= 3:
                print("Overfitting, stopping...")
                break

    def predict_for_user(self, user_idx, n_manga, top_k=10):
        self.eval()
        with torch.no_grad():
            all_manga_idx = torch.arange(n_manga)
            user_idx_repeated = torch.full((n_manga,), user_idx)
            xs = torch.stack([all_manga_idx, user_idx_repeated], dim=1)

            predictions = self(xs)

            top_k_values, top_k_indices = torch.topk(predictions, top_k)

            return top_k_indices.numpy(), top_k_values.numpy()


if __name__ == '__main__':
    #user df combining and cleaning
    manga_df = pd.read_csv('manga_dataset.csv')
    user_df = pd.read_csv('user_dataset.csv')
    user_df_2 = pd.read_csv('user_dataset_extras.csv')
    user_df = pd.concat([user_df, user_df_2], ignore_index=True)
    manga_db = list(manga_df['manga_id'])

    manga_db = set(list(manga_df['manga_id']))
    rated_manga = set(list(user_df['manga_id']))
    ids_to_remove = list(rated_manga - manga_db)

    user_df = user_df[~user_df['manga_id'].isin(ids_to_remove)]

    users_total = list(user_df['username'])
    user_counts = {}
    for user in users_total:
        user_counts[user] = user_counts.get(user, 0) + 1
    users_to_remove = [u for u, count in user_counts.items() if count < 10]

    user_df = user_df[~user_df['username'].isin(users_to_remove)]

    #creating indices
    all_users = sorted(user_df['username'].unique())
    all_manga = sorted(user_df['manga_id'].unique())

    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    manga_to_idx = {manga: idx for idx, manga in enumerate(all_manga)}

    #train test split
    user_df_temp, user_df_test = train_test_split(
        user_df,
        test_size=0.15,
        shuffle=True
    )

    user_df_train, user_df_val = train_test_split(
        user_df_temp,
        test_size=0.176,
        shuffle=True
    )

    #map
    for df in [user_df_train, user_df_val, user_df_test]:
        df.loc[:, 'manga_idx'] = df['manga_id'].map(manga_to_idx)
        df.loc[:, 'user_idx'] = df['username'].map(user_to_idx)

    #create tensors
    xs_train = torch.tensor(user_df_train[['manga_idx', 'user_idx']].values)
    ys_train = torch.tensor(user_df_train['user_score'].values, dtype=torch.float64)

    xs_val = torch.tensor(user_df_val[['manga_idx', 'user_idx']].values)
    ys_val = torch.tensor(user_df_val['user_score'].values, dtype=torch.float64)

    xs_test = torch.tensor(user_df_test[['manga_idx', 'user_idx']].values)
    ys_test = torch.tensor(user_df_test['user_score'].values, dtype=torch.float64)

    n_manga = len(all_manga)
    n_users = len(all_users)

    model = MatrixFactorisation(n_manga, n_users)

    model.train_loop(xs_train, ys_train, xs_val, ys_val, epochs=10000, lr=1e-3, decay=1e-3)

    #eval
    model.eval()

    #MSE Accuracy
    with torch.no_grad():
        test_preds = model(xs_test)
        loss = nn.MSELoss()
        error = loss(test_preds, ys_test)
        print(f"MSE Error: {error}")

    #Precision & Recall
    test_set_users = user_df_test['user_idx'].unique()
    k = 10
    relevance_threshold = 7
    with torch.no_grad():
        precisions = []
        recalls = []
        for user_idx in test_set_users:
            user_ratings = user_df_test[user_df_test['user_idx'] == user_idx].copy()
            relevant_items = user_ratings[user_ratings['user_score'] >= relevance_threshold]['manga_idx'].values

            if len(relevant_items) == 0:
                continue

            train_items = set(user_df_train[user_df_train['user_idx'] == user_idx]['manga_idx'].values)

            all_predictions, all_scores = model.predict_for_user(user_idx, n_manga, top_k=n_manga)

            predicted_topk = []
            for manga_idx in all_predictions:
                if manga_idx not in train_items:
                    predicted_topk.append(manga_idx)
                if len(predicted_topk) >= k:
                    break

            tp = len(set(predicted_topk) & set(relevant_items))
        
            precision = tp / k if k > 0 else 0
            recall = tp / len(relevant_items) if len(relevant_items) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)

        mean_precisions = np.mean(precisions)
        mean_recall = np.mean(recalls)
        print(f"Mean Average Precision: {mean_precisions}")
        print(f"Mean Average Recall: {mean_recall}")

    #Save and export
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'manga_to_idx': manga_to_idx,
        'user_to_idx': user_to_idx,
        'n_manga': n_manga,
        'n_users': n_users
    }, 'model.pth')
    print("\nSaved")