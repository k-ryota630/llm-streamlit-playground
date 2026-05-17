import os
import hmac
import streamlit as st

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


def check_password() -> bool:
    """シークレットに格納した APP_PASSWORD と照合する基本的なゲート。"""
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error(
            "APP_PASSWORD が設定されていません。Streamlit Cloud のシークレットを確認してください。"
        )
        st.stop()

    def password_entered():
        entered = st.session_state.get("password", "")
        # タイミング攻撃を避けるため定数時間比較を使う
        if hmac.compare_digest(str(entered), str(expected)):
            st.session_state["password_correct"] = True
            # 入力したパスワードをセッションに残さない
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "パスワード", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("パスワードが違います。")
    return False


MODEL_NAME = "gpt-5.2-pro"


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    # gpt-5.2-pro は応答に数分かかることがあるため、タイムアウトを長めに設定する
    return OpenAI(api_key=api_key, timeout=900.0)


def main():
    st.set_page_config(page_title="My Great ChatGPT (GPT-5.2 Pro)", page_icon="🤗")

    # --- 認証ゲート（通過するまで以降を実行しない） ---
    if not check_password():
        st.stop()

    st.header("My Great ChatGPT 5.2 Pro 🤗")

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OPENAI_API_KEY が設定されていません。Streamlit Cloud のシークレットを確認してください。"
        )
        st.stop()

    client = get_client(api_key)

    # --- サイドバー: Pro モデルは temperature 非対応のため reasoning effort を選ぶ ---
    reasoning_effort = st.sidebar.selectbox(
        "Reasoning effort（推論の強さ）",
        options=["medium", "high", "xhigh"],
        index=0,
        help="強くするほど精度は上がりやすい一方、応答時間とコストが増えます。",
    )

    system_prompt = "You are a helpful assistant."

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

        with st.spinner("GPT-5.2 Pro is thinking...（数分かかる場合があります）"):
            try:
                # Responses API は role 付きメッセージの配列を input として受け取れる
                input_messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    if msg["role"] in ("user", "assistant"):
                        input_messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )

                response = client.responses.create(
                    model=MODEL_NAME,
                    input=input_messages,
                    reasoning={"effort": reasoning_effort},
                )

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
