import streamlit as st
from backend.core import run_llm
from streamlit_chat import message
from typing import Set

# 사이드바에 사용자 정보 추가
with st.sidebar:
    st.title("사용자 정보")

    # 기본 프로필 이미지 URL (예시)
    default_profile_img = (
        # "https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y"
        "https://scontent-ssn1-1.xx.fbcdn.net/v/t39.30808-1/463744063_9332921913476096_7328666605000821833_n.jpg?stp=dst-jpg_s200x200_tt6&_nc_cat=105&ccb=1-7&_nc_sid=e99d92&_nc_ohc=cie4_GyAqeEQ7kNvgEb_eEX&_nc_zt=24&_nc_ht=scontent-ssn1-1.xx&_nc_gid=AGd4bL2diH3VtjgMtrGMom2&oh=00_AYAZZlSKm63kx-6XHa-l34DeHhqEdaqsSnC5qZTROQZMJw&oe=678EE630"
    )

    # 프로필 이미지
    st.image(default_profile_img, width=100)

    # 사용자 정보 표시
    # user_name = st.text_input("이름", placeholder="이름을 입력하세요")
    # user_email = st.text_input("이메일", placeholder="이메일을 입력하세요")
    st.text("이름: 박원석")
    st.text("이메일: amurorei2000@gmail.com")

    # 구분선
    st.divider()

# 메인 영역
st.header("LangChain - 문서 검색 봇")

prompt = st.text_input("Prompt", placeholder="프롬프트 작성...")

# streamlit 세션 상태에 채팅 히스토리 저장용 리스트 추가
if (
    "user_prompt_history" not in st.session_state
    and "chat_answers_history" not in st.session_state
    and "chat_history"
):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []
    st.session_state["chat_history"] = []


# metadata의 출처(source) 정리 포맷팅 함수
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    else:
        sources_list = list(source_urls)
        sources_list.sort()
        sources_string = "sources:\n"
        for i, sources in enumerate(sources_list):
            sources_string += f"{i + 1}. {sources}\n"
        return sources_string


if prompt:
    with st.spinner("답변을 생성 중입니다..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        result = generated_response["result"]
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        # 화면 출력용 답변 포맷팅
        formatted_response = f"{result} \n\n {create_sources_string(sources)}"

        # 질문과 답변 저장
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))


if st.session_state["chat_answers_history"]:
    for i, (generated_response, user_query) in enumerate(
        zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        )
    ):
        # 출력
        message(user_query, is_user=True, key=f"user_query_{i}")
        message(generated_response, is_user=False, key=f"ai_answer_{i}")
