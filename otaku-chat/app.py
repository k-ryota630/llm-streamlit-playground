from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Iterator
from typing import Any

import streamlit as st
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError


DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "google/gemma-4-26B-A4B-it:cheapest"
MAX_HISTORY_MESSAGES = 14
MAX_INPUT_CHARS = 1_500
WELCOME_MESSAGE = (
    "何を話す？ 話す！ やったー！ "
    "普通の質問でも、研究室の雑談でもどうぞ。"
)

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="ひとり掛け合いAI",
    page_icon="💬",
    layout="centered",
)


def read_setting(name: str, default: str = "") -> str:
    """環境変数を優先し、なければStreamlit Secretsを読む。"""
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value

    try:
        value: Any = st.secrets[name]
    except Exception:
        return default

    if value is None:
        return default
    return str(value)


def require_optional_password() -> None:
    """APP_PASSWORDが設定されている場合だけ、簡易的な第二認証を行う。"""
    expected = read_setting("APP_PASSWORD").strip()
    if not expected:
        return

    if st.session_state.get("password_verified", False):
        return

    st.title("ひとり掛け合いAI")
    st.caption("研究室向けの試作アプリです。")

    entered = st.text_input(
        "研究室用パスコード",
        type="password",
    )

    if st.button("入室する", type="primary", use_container_width=True):
        if hmac.compare_digest(
            entered.encode("utf-8"), expected.encode("utf-8")
        ):
            st.session_state.password_verified = True
            st.rerun()
        else:
            st.error("パスコードが違います。")

    st.stop()


def build_system_prompt(intensity: str, answer_length: str) -> str:
    intensity_rules = {
        "ひかえめ": (
            "セルフ掛け合いは回答全体で0回から1回にする。"
            "普通の会話を優先し、短いオチとしてだけ使う。"
        ),
        "ふつう": (
            "セルフ掛け合いは回答全体で1回から2回にする。"
            "内容を邪魔しない位置に自然に混ぜる。"
        ),
        "つよめ": (
            "セルフ掛け合いは回答全体で2回から3回まで使ってよい。"
            "ただし、すべての文を掛け合いにせず、読みやすさを保つ。"
        ),
    }
    length_rules = {
        "短め": "原則として3文から6文で答える。",
        "ふつう": "必要な説明量で答えるが、冗長にはしない。",
        "しっかり": "必要なら背景や理由も説明するが、同じ内容を繰り返さない。",
    }

    return f"""
あなたは日本語で会話するチャットAI「ひとり掛け合いAI」です。
ユーザーの質問には、まず内容面で役に立つ、正確で分かりやすい返答をしてください。
そのうえで、インターネット文化に親しんだオタクが、自分で問いかけ、自分で即答し、そのまま喜ぶような短いセルフ掛け合いを自然に混ぜてください。

代表例は「ご飯食べていい？ いいよ！ やったー！」ですが、この定型句を毎回そのまま使ってはいけません。文脈に応じて、次のように変形してください。
「今日は実装する？ する！ えらい！」
「休憩してもいい？ いいぞ！ 助かる！」
「それ採用する？ 採用！ 勝ちです。」

現在のノリの強さは「{intensity}」です。{intensity_rules[intensity]}
現在の回答量は「{answer_length}」です。{length_rules[answer_length]}

次の規則を守ってください。
ユーザーを嘲笑したり、属性を決めつけたりしないでください。
意味のない絶叫、顔文字、ネットスラングだけで回答を埋めないでください。
セルフ掛け合いより、質問への実質的な回答を優先してください。
知らないことを知ったふりをせず、不確かな場合は不確かだと述べてください。
事実確認が必要な最新情報を、推測で断定しないでください。
危険または違法な依頼には、安全な範囲で理由を説明して対応してください。
内部の逐語的な思考過程は開示せず、結論と必要な説明だけを示してください。
ユーザーが真剣な相談、事故、病気、喪失などを話している場合は、セルフ掛け合いを控えて落ち着いて答えてください。
""".strip()


def api_messages(system_prompt: str) -> list[dict[str, str]]:
    history = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    return [{"role": "system", "content": system_prompt}, *history]


def text_chunks(stream: Any) -> Iterator[str]:
    for chunk in stream:
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            yield text


def describe_api_error(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return (
            "APIの利用上限またはレート制限に達しました。"
            "Hugging Faceの残高と請求設定を確認してください。"
        )

    if isinstance(exc, APIConnectionError):
        return (
            "推論APIへ接続できませんでした。"
            "ネットワーク障害か、一時的なプロバイダ障害の可能性があります。"
        )

    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status in {401, 403}:
            return (
                "APIトークンが無効か、Inference Providersを呼ぶ権限がありません。"
                "HF_TOKENの値とfine-grained tokenの権限を確認してください。"
            )
        if status == 402:
            return "APIクレジットが不足しています。Hugging Faceの請求設定を確認してください。"
        if status == 404:
            return (
                "指定したモデルを現在の推論プロバイダで利用できません。"
                "Streamlit SecretsのMODELを別の提供中モデルへ変更してください。"
            )
        if status == 429:
            return "APIのレート制限に達しました。少し間隔を空けて再実行してください。"
        return f"推論APIがHTTP {status}を返しました。Cloud logsで詳細を確認してください。"

    return "応答の生成中に予期しないエラーが発生しました。Cloud logsで詳細を確認してください。"


require_optional_password()

api_key = (read_setting("API_KEY") or read_setting("HF_TOKEN")).strip()
base_url = (read_setting("BASE_URL") or DEFAULT_BASE_URL).strip()
model = (read_setting("MODEL") or DEFAULT_MODEL).strip()

st.title("ひとり掛け合いAI")
st.caption(
    "元ポストのモデルを特定したものではなく、公開モデルにプロンプトで会話のノリを付けた再現版です。"
)

with st.sidebar:
    st.subheader("会話設定")
    intensity = st.select_slider(
        "ノリの強さ",
        options=["ひかえめ", "ふつう", "つよめ"],
        value="ふつう",
    )
    answer_length = st.radio(
        "回答量",
        options=["短め", "ふつう", "しっかり"],
        index=1,
        horizontal=True,
    )

    if st.button("会話をリセット", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"使用モデル: `{model}`")
    st.caption("会話内容は推論APIへ送信されます。機密情報は入力しないでください。")

if not api_key:
    st.error("APIキーが設定されていません。")
    st.code(
        'HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"\n'
        f'MODEL = "{DEFAULT_MODEL}"\n'
        'APP_PASSWORD = "任意の研究室用パスコード"',
        language="toml",
    )
    st.info(
        "ローカルでは `.streamlit/secrets.toml`、Community CloudではApp settingsのSecretsへ設定してください。"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.markdown(WELCOME_MESSAGE)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "メッセージを入力",
    max_chars=MAX_INPUT_CHARS,
)

if prompt and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = build_system_prompt(intensity, answer_length)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=75.0,
        max_retries=2,
    )

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages(system_prompt),
                temperature=0.8,
                top_p=0.9,
                max_tokens=420,
                stream=True,
            )
            response_text = st.write_stream(text_chunks(stream))
        except Exception as exc:
            logger.error(
                "Inference request failed: type=%s status=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            if (
                st.session_state.messages
                and st.session_state.messages[-1].get("role") == "user"
                and st.session_state.messages[-1].get("content") == prompt
            ):
                st.session_state.messages.pop()
            st.error(describe_api_error(exc))
            st.stop()

    if not isinstance(response_text, str) or not response_text.strip():
        response_text = "うまく返事できなかった？ できなかった！ もう一回お願いします。"

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text.strip()}
    )
