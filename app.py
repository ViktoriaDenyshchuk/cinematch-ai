
import streamlit as st
from agent import run_agent
from movie_tools import MOVIES

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f8fafc;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #ffffff !important;
}
[data-testid="stChatMessage"] {
    background: #161e33;
    border: 1px solid #28334e;
    border-radius: 16px;
}
[data-testid="stMetric"] {
    background: #161e33;
    padding: 16px;
    border-radius: 14px;
}
.stButton > button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🎬 CineMatch AI")
    st.caption("Your intelligent movie companion")

    st.divider()

    st.subheader("📊 MovieLens Dataset")
    st.metric("Фільмів у базі", f"{len(MOVIES):,}")
    st.metric(
        "Кількість оцінок",
        f"{MOVIES['rating_count'].sum():,}"
    )

    st.divider()

    st.markdown("**Модель:** Groq GPT-OSS 20B")
    st.markdown("**Дані:** MovieLens Latest Small")
    st.markdown("**Інструменти:** Search & Compare")

    st.divider()

    if st.button("🗑️ Очистити чат", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🎬 CineMatch AI")
st.markdown(
    "### Знайди свій наступний улюблений фільм"
)
st.caption(
    "Шукай, порівнюй та отримуй рекомендації "
    "на основі реальних рейтингів MovieLens."
)

st.divider()

st.markdown("#### ✨ Спробуй один із запитів")

examples = [
    "Порадь 5 комедій із рейтингом від 4.",
    "Порівняй Toy Story та Jumanji.",
    "Знайди фантастичні фільми після 2000 року.",
    "Порівняй Titanic та Avatar."
]

cols = st.columns(2)
selected_query = None

for i, example in enumerate(examples):
    with cols[i % 2]:
        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):
            selected_query = example

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("logs"):
            with st.expander("🔍 Журнал дій AI-агента"):
                for log in message["logs"]:
                    st.markdown(
                        f"**Крок {log['step']}: "
                        f"`{log['tool']}`**"
                    )
                    st.json(log)

query = st.chat_input(
    "Наприклад: Порадь 5 комедій із високим рейтингом..."
)

if selected_query:
    query = selected_query

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🎬 Шукаю найкращі фільми..."):
            try:
                result = run_agent(query)

                answer = result["answer"]
                logs = result["logs"]

                st.markdown(answer)

                if logs:
                    with st.expander("🔍 Журнал дій AI-агента"):
                        for log in logs:
                            st.markdown(
                                f"**Крок {log['step']}: "
                                f"`{log['tool']}`**"
                            )
                            st.json(log)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "logs": logs
                })

            except Exception as exc:
                st.error(
                    "Не вдалося отримати відповідь від AI. "
                    "Перевір підключення до Groq."
                )
                with st.expander("Технічні деталі"):
                    st.code(str(exc))
