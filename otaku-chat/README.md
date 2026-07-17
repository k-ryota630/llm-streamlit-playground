# ひとり掛け合いAI

「ご飯食べていい？ いいよ！ やったー！」のように、一人で問いかけと返答を完結させる会話のノリを、公開モデルとプロンプトで再現するStreamlitアプリです。

元ポストで使われたモデル自体は特定できていません。このリポジトリは、そのモデルを複製したものではありません。Hugging Face Inference Providersで提供中の汎用モデルへ、システムプロンプトと例示を与えて挙動を再現します。

既定モデルは、Google DeepMindの命令調整済みモデル `google/gemma-4-26B-A4B-it` です。Hugging Face Routerでは `:cheapest` を付け、利用可能なプロバイダのうち出力単価が低い経路を選びます。

## 構成

ブラウザからStreamlit Community Cloudへ接続し、StreamlitアプリがHugging Face Router経由で推論プロバイダを呼びます。モデル本体はStreamlit Community Cloudへ同梱しません。

会話履歴はStreamlitの `st.session_state` に保持します。データベースには保存しないため、ブラウザセッションの終了、アプリの再起動、休止からの復帰などで履歴が消える場合があります。入力内容はStreamlit Community Cloudと推論プロバイダを通過するため、未公開の研究データ、個人情報、認証情報は入力しないでください。

## ローカル起動

Python 3.12を推奨します。プロジェクトのルートで次を実行します。

```bash
python -m venv .venv
```

Windows PowerShellでは次のように仮想環境を有効化します。

```powershell
.venv\Scripts\Activate.ps1
```

macOSまたはLinuxでは次のように有効化します。

```bash
source .venv/bin/activate
```

依存関係を入れます。

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Secretsの見本をコピーします。

Windows PowerShellでは次を実行します。

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

macOSまたはLinuxでは次を実行します。

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

`.streamlit/secrets.toml` の `HF_TOKEN` を実際の値に置き換えます。トークンにはHugging Faceの `Make calls to Inference Providers` 権限が必要です。

起動します。

```bash
streamlit run app.py
```

## Streamlit Community Cloudへのデプロイ

GitHubでprivateリポジトリを作り、この一式をpushします。`.streamlit/secrets.toml` はpushしません。`.gitignore` により除外されます。

Streamlit Community Cloudで新しいアプリを作り、privateリポジトリ、ブランチ、`app.py` を指定してデプロイします。

アプリのSettingsからSecretsを開き、次の内容を設定します。

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
MODEL = "google/gemma-4-26B-A4B-it:cheapest"
BASE_URL = "https://router.huggingface.co/v1"
APP_PASSWORD = "研究室内で共有する任意のパスコード"
```

Sharingでは `Only specific people can view this app` を選び、当日利用する研究室メンバーのメールアドレスだけを招待します。privateリポジトリからデプロイしただけで満足せず、公開範囲を画面で確認してください。

Community Cloudのprivate appは現在1個までです。すでにprivate appを使っている場合は、既存アプリの扱いを先に決める必要があります。

## Hugging Faceの費用

Hugging Faceの無料ユーザーに付くInference Providersの月間クレジットは少額です。短い動作確認には使えても、研究室で何人も連続利用すると尽きる可能性があります。デモ前日に残高を確認し、必要なら追加クレジットを購入してください。

モデル名末尾の `:cheapest` は、そのモデルを提供するプロバイダのうち、低価格側を選ぶための指定です。提供状況が変わって404になる場合は、Secretsの `MODEL` を次の候補へ変更してください。

```toml
MODEL = "google/gemma-4-31B-it:cheapest"
```

モデルを変えても `app.py` の変更は不要です。

## アクセス制御の意味

この構成は「招待した人だけがURLを開ける」という意味で非公開にできます。ただし、処理を研究室LAN内だけで完結させる構成ではありません。Community Cloudは米国内でホストされ、入力はインターネット経由で外部推論APIへ送られます。

未公開研究データを扱う場合は、Streamlit Community Cloudを使わず、研究室LAN内で `streamlit run app.py --server.address 0.0.0.0` を実行し、研究室のOllamaまたはLiteLLMへ `BASE_URL` を向ける構成へ変更してください。

## 動作確認用の入力例

```text
今日の昼ごはんをカレーにしてもいい？
```

```text
研究室の発表資料を作る気が起きません。最初の10分で何をすればいい？
```

```text
Pythonのリスト内包表記を初心者向けに説明して。
```

## 実装上の注意

Streamlitはウィジェット操作のたびにスクリプトを上から再実行します。通常のローカル変数だけで会話履歴を持つと、入力のたびに初期化されます。そのため、このアプリでは `st.session_state.messages` に履歴を入れています。

`APP_PASSWORD` は任意の第二認証です。これだけでpublic appを安全なprivate appへ変えられるわけではありません。Community Cloudのviewer allow-listを主なアクセス制御として使ってください。
