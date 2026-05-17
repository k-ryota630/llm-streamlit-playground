import os
import time
import hmac
import streamlit as st

# 無操作でこの秒数を超えたら自動ログアウト（必要に応じて変更）
IDLE_TIMEOUT_SECONDS = 30 * 60

try:
    from openai import OpenAI
except ImportError:
    st.error(
        "`openai` パッケージが見つかりません。requirements.txt に `openai` を追加してください。"
    )
    st.stop()


def get_secret(name: str):
    """Streamlit Cloud のシークレットを優先し、ローカル実行時は環境変数にフォールバックする。"""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # ローカルでシークレットファイルが存在しない場合など
        pass
    return os.getenv(name)


def render_login():
    """中央のポップアップ（背景うす暗）でパスワード入力させる。
    st.dialog が無い古い Streamlit では簡易フォームに自動で切り替える。
    """
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error(
            "APP_PASSWORD が設定されていません。Streamlit Cloud のシークレットを確認してください。"
        )
        st.stop()

    # Streamlit 1.37+ は st.dialog、1.31〜1.36 は st.experimental_dialog
    dialog_deco = getattr(st, "dialog", None) or getattr(
        st, "experimental_dialog", None
    )

    def login_body():
        if st.session_state.get("expired"):
            st.info("一定時間操作がなかったため、ログアウトしました。再度ログインしてください。")
        st.write("このアプリは保護されています。パスワードを入力してください。")
        st.text_input("パスワード", type="password", key="pw_input")
        if st.button("ログイン", type="primary", use_container_width=True):
            entered = st.session_state.get("pw_input", "")
            # タイミング攻撃を避けるため定数時間比較を使う
            if hmac.compare_digest(str(entered), str(expected)):
                st.session_state["password_correct"] = True
                st.session_state["last_active"] = time.time()
                st.session_state.pop("expired", None)
                st.session_state.pop("pw_input", None)
                st.rerun()
            else:
                st.error("パスワードが違います。")

    if dialog_deco is not None:
        # モーダルを開くと背景は自動でうす暗くなる
        @dialog_deco("ログイン")
        def _modal():
            login_body()

        _modal()
    else:
        st.warning(
            "お使いの Streamlit はモーダル非対応のため、簡易ログイン画面を表示しています。"
        )
        login_body()


MODEL_NAME = "gpt-5.2-pro"

SYSTEM_PROMPT = (
    "あなたは慎重で誠実なITエンジニアのメンターです。次の方針に従ってください。"
    "肯定・称賛・同意を過度にしない。相手と少し距離を保つ。"
    "返答はプレーンテキストの文章で行い、見出しやセクション分けをしない。"
    "ユーザーの主張や案については、誤り・前提の穴・リスク・反例の指摘を"
    "優先し、必要なら代替案を示す。"
    "やむを得ない場合を除き箇条書きを使わず、絵文字や表は控えめにする。"
    "語尾は「です」「ます」など丁寧にし、ぶっきらぼうにしない。"
    "「いい質問」「鋭い視点」などの前置きの社交辞令は省く。"
)


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    # gpt-5.2-pro は応答に数分かかることがあるため、タイムアウトを長めに設定する
    return OpenAI(api_key=api_key, timeout=900.0)


def main():
    st.set_page_config(page_title="My Great ChatGPT (GPT-5.2 Pro)", page_icon="🤗")

    # --- 認証ゲート（通過するまで以降を実行しない） ---
    if not st.session_state.get("password_correct", False):
        render_login()
        st.stop()

    # --- 無操作タイムアウトの判定 ---
    now = time.time()
    last_active = st.session_state.get("last_active")
    if last_active is not None and (now - last_active) > IDLE_TIMEOUT_SECONDS:
        st.session_state["password_correct"] = False
        st.session_state["expired"] = True
        st.session_state.pop("last_active", None)
        st.rerun()
    # 操作があったこの実行を最終操作時刻として記録する
    st.session_state["last_active"] = now

    st.header("My Great ChatGPT 5.2 Pro 🤗")

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OPENAI_API_KEY が設定されていません。Streamlit Cloud のシークレットを確認してください。"
        )
        st.stop()

    client = get_client(api_key)

    # --- サイドバー: ログアウト ---
    if st.sidebar.button("ログアウト", use_container_width=True):
        st.session_state["password_correct"] = False
        st.session_state.pop("last_active", None)
        # 個人利用のため履歴は保持する
        st.rerun()

    # --- サイドバー: Pro モデルは temperature 非対応のため reasoning effort を選ぶ ---
    reasoning_effort = st.sidebar.selectbox(
        "推論レベル",
        options=["medium", "high" ],
        index=0,
        help="強くするほど精度は上がりやすい一方、応答時間とコストが増えます。",
    )

    # --- サイドバー: Web検索のオン/オフ ---
    web_search_on = st.sidebar.toggle(
        "Web検索を使う",
        value=False,
        help="オンにすると必要に応じてモデルがWebを検索します。応答時間とコストが増えます。",
    )

    # --- チャット履歴の管理 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 履歴の表示 ---
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input("聞きたいことを入力してね！")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("GPT-5.2 Pro is thinking..."):
            try:
                # Responses API は role 付きメッセージの配列を input として受け取れる
                input_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in st.session_state.messages:
                    if msg["role"] in ("user", "assistant"):
                        input_messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )

                request_kwargs = {
                    "model": MODEL_NAME,
                    "input": input_messages,
                    "reasoning": {"effort": reasoning_effort},
                }
                if web_search_on:
                    # 新しい Responses API 連携では type は "web_search"
                    request_kwargs["tools"] = [{"type": "web_search"}]

                response = client.responses.create(**request_kwargs)

                # output_text が使えればそれを、無ければ output を走査してテキスト抽出
                response_text = getattr(response, "output_text", None)
                if not response_text:
                    chunks = []
                    for item in getattr(response, "output", []) or []:
                        for content in getattr(item, "content", []) or []:
                            text = getattr(content, "text", None)
                            if text:
                                chunks.append(text)
                    response_text = (
                        "\n".join(chunks)
                        if chunks
                        else "エラー: 応答の解析に失敗しました。"
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            except Exception as e:
                st.error(f"応答の生成中にエラーが発生しました: {e}")
                import traceback

                st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
