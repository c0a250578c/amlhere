# Firebase デプロイ手順

## 前提条件

1. Node.js (v18以上) がインストールされていること
2. Firebase CLI がインストールされていること
   ```bash
   npm install -g firebase-tools
   ```

## ステップ1: 認証（あなたが実行）

```bash
firebase login
```

ブラウザが開くので、Googleアカウントでログインしてください。

## ステップ2: Firebaseプロジェクトの作成

```bash
firebase projects:create amlhere-app
```

または、Firebase Console (https://console.firebase.google.com) で新しいプロジェクトを作成して、プロジェクトIDを控えてください。

## ステップ3: .firebaserc の更新（プロジェクトIDを設定）

`.firebaserc` ファイルを編集して、実際のプロジェクトIDを設定してください：

```json
{
  "projects": {
    "default": "あなたのプロジェクトID"
  }
}
```

## ステップ4: Blazeプランの有効化（必須）

Firebase Functionsを使用するには、Blaze（従量課金）プランが必要です：

1. https://console.firebase.google.com にアクセス
2. プロジェクトを選択
3. 「Sparkプランを変更」または「Blazeプランにアップグレード」をクリック
4. 請求先アカウントを設定

## ステップ5: 環境変数の設定

```bash
firebase functions:config:set gemini.key="YOUR_GEMINI_API_KEY"
```

または、Firebase Consoleで設定：
1. Functions > 環境変数
2. `GEMINI_API_KEY` を追加

## ステップ6: デプロイ

```bash
# 依存関係をインストール
cd functions
pip install -r requirements.txt
cd ..

# デプロイ実行
firebase deploy
```

## デプロイ後の確認

デプロイ後、以下のURLで確認できます：

- **Webアプリ**: `https://amlhere-app.web.app`
- **API**:
  - `https://us-central1-amlhere-app.cloudfunctions.net/chat`
  - `https://us-central1-amlhere-app.cloudfunctions.net/getMemories`
  - など...

## トラブルシューティング

### デプロイに失敗する場合

1. **請求先アカウントが設定されていない** → Blazeプランにアップグレード
2. **APIが有効になっていない** → Google Cloud Consoleで Cloud Functions API, Cloud Build API を有効化
3. **権限エラー** → `firebase logout` → `firebase login` で再認証

### 関数のログを確認

```bash
firebase functions:log
```

## 無料枠について

- **Firebase Functions**: 月あたり200万回の呼び出し無料
- **Firebase Hosting**: 1GBのストレージ、10GB/月の転送量無料
- **Cloud Firestore**（使用しない場合）: 適用されません（SQLite使用）

詳細: https://firebase.google.com/pricing
